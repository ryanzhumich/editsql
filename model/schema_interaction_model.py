""" Class for the Sequence to sequence model for ATIS."""

import torch
import torch.nn.functional as F
from . import torch_utils

import data_util.snippets as snippet_handler
import data_util.sql_util
import data_util.vocabulary as vocab
from data_util.vocabulary import EOS_TOK, UNK_TOK
import data_util.tokenizers

from .token_predictor import construct_token_predictor
from .attention import Attention
from .model import ATISModel, encode_snippets_with_states, get_token_indices
from data_util.utterance import ANON_INPUT_KEY

from .encoder import Encoder
from .decoder import SequencePredictorWithSchema

import data_util.atis_batch


LIMITED_INTERACTIONS = {"raw/atis2/12-1.1/ATIS2/TEXT/TRAIN/SRI/QS0/1": 22,
                        "raw/atis3/17-1.1/ATIS3/SP_TRN/MIT/8K7/5": 14,
                        "raw/atis2/12-1.1/ATIS2/TEXT/TEST/NOV92/770/5": -1}

END_OF_INTERACTION = {"quit", "exit", "done"}


class SchemaInteractionATISModel(ATISModel):
    """ Interaction ATIS model, where an interaction is processed all at once.
    """

    def __init__(self,
                 params,
                 input_vocabulary,
                 output_vocabulary,
                 output_vocabulary_schema,
                 anonymizer):
        ATISModel.__init__(
            self,
            params,
            input_vocabulary,
            output_vocabulary,
            output_vocabulary_schema,
            anonymizer)

        self.token_predictor = construct_token_predictor(params,
                                                         output_vocabulary,
                                                         self.utterance_attention_key_size,
                                                         self.schema_attention_key_size,
                                                         self.final_snippet_size,
                                                         anonymizer)


        decoder_input_size = params.output_embedding_size + self.utterance_attention_key_size

        self.decoder = SequencePredictorWithSchema(params, decoder_input_size, self.output_embedder, self.column_name_token_embedder, self.token_predictor)


    def predict_turn(self,
                     utterance_final_state,
                     input_hidden_states,
                     schema_states,
                     max_generation_length,
                     gold_query=None,
                     snippets=None,
                     input_sequence=None,
                     input_schema=None,
                     feed_gold_tokens=False,
                     training=False):
        """ Gets a prediction for a single turn -- calls decoder and updates loss, etc.

        TODO:  this can probably be split into two methods, one that just predicts
            and another that computes the loss.
        """
        predicted_sequence = []
        fed_sequence = []
        loss = None
        token_accuracy = 0.

        if feed_gold_tokens:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           schema_states,
                                           max_generation_length,
                                           gold_sequence=gold_query,
                                           input_sequence=input_sequence,
                                           input_schema=input_schema,
                                           snippets=snippets,
                                           dropout_amount=self.dropout)

            all_scores = [step.scores for step in decoder_results.predictions]
            all_alignments = [step.aligned_tokens for step in decoder_results.predictions]

            # Compute the loss
            gold_sequence = gold_query

            loss = torch_utils.compute_loss(gold_sequence, all_scores, all_alignments, get_token_indices)
            if not training:
                predicted_sequence = torch_utils.get_seq_from_scores(all_scores, all_alignments)
                token_accuracy = torch_utils.per_token_accuracy(gold_sequence, predicted_sequence)
            fed_sequence = gold_sequence
        else:
            decoder_results = self.decoder(utterance_final_state,
                                           input_hidden_states,
                                           schema_states,
                                           max_generation_length,
                                           input_sequence=input_sequence,
                                           input_schema=input_schema,
                                           snippets=snippets,
                                           dropout_amount=self.dropout)
            predicted_sequence = decoder_results.sequence
            fed_sequence = predicted_sequence

        decoder_states = [pred.decoder_state for pred in decoder_results.predictions]

        # fed_sequence contains EOS, which we don't need when encoding snippets.
        # also ignore the first state, as it contains the BEG encoding.

        for token, state in zip(fed_sequence[:-1], decoder_states[1:]):
            if snippet_handler.is_snippet(token):
                snippet_length = 0
                for snippet in snippets:
                    if snippet.name == token:
                        snippet_length = len(snippet.sequence)
                        break
                assert snippet_length > 0
                decoder_states.extend([state for _ in range(snippet_length)])
            else:
                decoder_states.append(state)

        return (predicted_sequence,
                loss,
                token_accuracy,
                decoder_states,
                decoder_results)

    def encode_schema_bow_simple(self, input_schema):
        schema_states = []
        for column_name in input_schema.column_names_embedder_input:
            schema_states.append(input_schema.column_name_embedder_bow(column_name, surface_form=False, column_name_token_embedder=self.column_name_token_embedder))
        input_schema.set_column_name_embeddings(schema_states)
        return schema_states

    def get_query_token_embedding(self, output_token, input_schema):
        if input_schema:
            if not (self.output_embedder.in_vocabulary(output_token) or input_schema.in_vocabulary(output_token, surface_form=True)):
                output_token = 'value'
            if self.output_embedder.in_vocabulary(output_token):
                output_token_embedding = self.output_embedder(output_token)
            else:
                output_token_embedding = input_schema.column_name_embedder(output_token, surface_form=True)
        else:
            output_token_embedding = self.output_embedder(output_token)
        return output_token_embedding

    def train_step(self, interaction, max_generation_length, snippet_alignment_probability=1.):
        """ Trains the interaction-level model on a single interaction.

        Inputs:
            interaction (Interaction): The interaction to train on.
            learning_rate (float): Learning rate to use.
            snippet_keep_age (int): Age of oldest snippets to use.
            snippet_alignment_probability (float): The probability that a snippet will
                be used in constructing the gold sequence.
        """
        # assert self.params.discourse_level_lstm

        losses = []
        total_gold_tokens = 0

        input_hidden_states = []
        input_sequences = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema:
            schema_states = self.encode_schema_bow_simple(input_schema)

        for utterance_index, utterance in enumerate(interaction.gold_utterances()):
            if interaction.identifier in LIMITED_INTERACTIONS and utterance_index > LIMITED_INTERACTIONS[interaction.identifier]:
                break

            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Get the gold query: reconstruct if the alignment probability is less than one
            if snippet_alignment_probability < 1.:
                gold_query = sql_util.add_snippets_to_query(
                    available_snippets,
                    utterance.contained_entities(),
                    utterance.anonymized_gold_query(),
                    prob_align=snippet_alignment_probability) + [vocab.EOS_TOK]
            else:
                gold_query = utterance.gold_query()

            # Encode the utterance, and update the discourse-level states
            if self.params.discourse_level_lstm:
                utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
            else:
                utterance_token_embedder = self.input_embedder
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                utterance_token_embedder,
                dropout_amount=self.dropout)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            # final_utterance_state[1][0] is the first layer's hidden states at the last time step (concat forward lstm and backward lstm)
            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            if len(gold_query) <= max_generation_length and len(previous_query) <= max_generation_length:
                prediction = self.predict_turn(final_utterance_state,
                                               utterance_states,
                                               schema_states,
                                               max_generation_length,
                                               gold_query=gold_query,
                                               snippets=snippets,
                                               input_sequence=flat_sequence,
                                               input_schema=input_schema,
                                               feed_gold_tokens=True,
                                               training=True)
                loss = prediction[1]
                decoder_states = prediction[3]
                total_gold_tokens += len(gold_query)
                losses.append(loss)
            else:
                # Break if previous decoder snippet encoding -- because the previous
                # sequence was too long to run the decoder.
                if self.params.previous_decoder_snippet_encoding:
                    break
                continue

            torch.cuda.empty_cache()

        if losses:
            average_loss = torch.sum(torch.stack(losses)) / total_gold_tokens

            # Renormalize so the effect is normalized by the batch size.
            normalized_loss = average_loss
            if self.params.reweight_batch:
                normalized_loss = len(losses) * average_loss / float(self.params.batch_size)

            normalized_loss.backward()
            self.trainer.step()
            self.zero_grad()

            loss_scalar = normalized_loss.item()
        else:
            loss_scalar = 0.

        return loss_scalar

    def predict_with_predicted_queries(self, interaction, max_generation_length, syntax_restrict=True):
        """ Predicts an interaction, using the predicted queries to get snippets."""
        # assert self.params.discourse_level_lstm

        syntax_restrict=False

        predictions = []

        input_hidden_states = []
        input_sequences = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []

        if input_schema:
            schema_states = self.encode_schema_bow_simple(input_schema)

        interaction.start_interaction()
        while not interaction.done():
            utterance = interaction.next_utterance()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            input_sequence = utterance.input_sequence()

            if self.params.discourse_level_lstm:
                utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
            else:
                utterance_token_embedder = self.input_embedder
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                utterance_token_embedder)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            results = self.predict_turn(final_utterance_state,
                                        utterance_states,
                                        schema_states,
                                        max_generation_length,
                                        input_sequence=flat_sequence,
                                        input_schema=input_schema,
                                        snippets=snippets)

            predicted_sequence = results[0]
            predictions.append(results)

            # Update things necessary for using predicted queries
            anonymized_sequence = utterance.remove_snippets(predicted_sequence)
            if EOS_TOK in anonymized_sequence:
                anonymized_sequence = anonymized_sequence[:-1] # Remove _EOS
            else:
                anonymized_sequence = ['select', '*', 'from', 't1']

            if not syntax_restrict:
                utterance.set_predicted_query(interaction.remove_snippets(predicted_sequence))
                if input_schema:
                    # on SParC
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=True)
                else:
                    # on ATIS
                    interaction.add_utterance(utterance, anonymized_sequence, previous_snippets=utterance.snippets(), simple=False)
            else:
                utterance.set_predicted_query(utterance.previous_query())
                interaction.add_utterance(utterance, utterance.previous_query(), previous_snippets=utterance.snippets())

        return predictions


    def predict_with_gold_queries(self, interaction, max_generation_length, feed_gold_query=False):
        """ Predicts SQL queries for an interaction.

        Inputs:
            interaction (Interaction): Interaction to predict for.
            feed_gold_query (bool): Whether or not to feed the gold token to the
                generation step.
        """
        # assert self.params.discourse_level_lstm

        predictions = []

        input_hidden_states = []
        input_sequences = []

        decoder_states = []

        discourse_state = None
        if self.params.discourse_level_lstm:
            discourse_state, discourse_lstm_states = self._initialize_discourse_states()

        # Schema and schema embeddings
        input_schema = interaction.get_schema()
        schema_states = []
        if input_schema:
            schema_states = self.encode_schema_bow_simple(input_schema)


        for utterance in interaction.gold_utterances():
            input_sequence = utterance.input_sequence()

            available_snippets = utterance.snippets()
            previous_query = utterance.previous_query()

            # Encode the utterance, and update the discourse-level states
            if self.params.discourse_level_lstm:
                utterance_token_embedder = lambda token: torch.cat([self.input_embedder(token), discourse_state], dim=0)
            else:
                utterance_token_embedder = self.input_embedder
            final_utterance_state, utterance_states = self.utterance_encoder(
                input_sequence,
                utterance_token_embedder,
                dropout_amount=self.dropout)

            input_hidden_states.extend(utterance_states)
            input_sequences.append(input_sequence)

            num_utterances_to_keep = min(self.params.maximum_utterances, len(input_sequences))

            if self.params.discourse_level_lstm:
                _, discourse_state, discourse_lstm_states = torch_utils.forward_one_multilayer(self.discourse_lstms, final_utterance_state[1][0], discourse_lstm_states, self.dropout)

            if self.params.state_positional_embeddings:
                utterance_states, flat_sequence = self._add_positional_embeddings(input_hidden_states, input_sequences)
            else:
                flat_sequence = []
                for utt in input_sequences[-num_utterances_to_keep:]:
                    flat_sequence.extend(utt)

            snippets = None
            if self.params.use_snippets:
                if self.params.previous_decoder_snippet_encoding:
                    snippets = encode_snippets_with_states(available_snippets, decoder_states)
                else:
                    snippets = self._encode_snippets(previous_query, available_snippets, input_schema)

            prediction = self.predict_turn(final_utterance_state,
                                           utterance_states,
                                           schema_states,
                                           max_generation_length,
                                           gold_query=utterance.gold_query(),
                                           snippets=snippets,
                                           input_sequence=flat_sequence,
                                           input_schema=input_schema,
                                           feed_gold_tokens=feed_gold_query)

            decoder_states = prediction[3]
            predictions.append(prediction)

        return predictions