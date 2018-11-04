from modules import *
from torch.nn.functional import relu, tanh


class TextRNN(Module):
    """
    This  simple architecture  trains all data by using only the body of the post. It is a simple RNN
    where the final representation of a text is the output from the RNN of the last word. This last
    state passes from a MLP  giving as final output a scalar. In training, this output passes from a
    sigmoid in order to be transformed to probability.
    """
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers of the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "TextRNN"

        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.seq = SequentialModel([self.embedding_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class ProjectedTextRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=torch.tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param top_mlp_layers: number of layers of the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "ProjectedTextRNN"

        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers, init_size=self.embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)

        self.cell = CellLayer(is_gru, self.projection_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = LastState(self.cell.get_output_size(), self.decision_layer.get_input_size())
        self.seq = SequentialModel([self.embedding_layer, self.projection_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class AttentionTextRNN(Module):
    def __init__(self, embeddings,
                 trainable_embeddings=True,
                 embeddings_dropout=0.0,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param embeddings_dropout: dropout of the embeddings layer
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "AttentionTextRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.cell = CellLayer(is_gru, self.word_embedding_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class ProjectedAttentionTextRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=torch.tanh,
                 proj_mlp_dropout=0.5,
                 proj_size=50,
                 is_gru=True,
                 cell_hidden_size=128,
                 stacked_layers=1,
                 bidirectional=False,
                 att_mlp_layers=1,
                 att_mlp_dropout=0.5,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_size: hidden size of the projection layers
        :param is_gru: GRU cell type if true, otherwise LSTM
        :param cell_hidden_size: the cell size of the RNN
        :param stacked_layers: the number of stacked layers of the RNN
        :param bidirectional: boolean indicating if the cell is bidirectional
        :param att_mlp_layers: number of layers of the attention mlp
        :param att_mlp_dropout: dropout of the attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextRNN, self).__init__()
        self.input_list = ['text']
        self.name = "ProjectedAttentionTextRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)

        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers, init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_size, dropout=proj_mlp_dropout,
                                                     inner_activation=proj_mlp_activation, outer_activation=proj_mlp_activation)

        self.cell = CellLayer(is_gru, self.projection_layer.get_output_size(),
                              cell_hidden_size, bidirectional, stacked_layers)
        large_size = cell_hidden_size * 2 if bidirectional else cell_hidden_size
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.last_state = AttendedState(att_mlp_layers, large_size, att_mlp_dropout, relu)
        self.seq = SequentialModel([self.word_embedding_layer, self.projection_layer, self.cell, self.last_state, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x):
        out = self.seq(x)
        return out


class TextSummaryRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 text_is_gru=True,
                 text_cell_hidden_size=64,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 summary_is_gru=True,
                 summary_cell_hidden_size=32,
                 summary_stacked_layers=1,
                 summary_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the RNN for text representation
        :param text_stacked_layers: the number of stacked layers of the RNN for text representation
        :param text_bidirectional: boolean indicating if the RNN for text representation is bidirectional
        :param summary_is_gru: GRU cell type for the summary cell if true, otherwise LSTM
        :param summary_cell_hidden_size: the cell size of the RNN for summary representation
        :param summary_stacked_layers: the number of stacked layers of the RNN for summary representation
        :param summary_bidirectional: boolean indicating if the RNN for summary representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(TextSummaryRNN, self).__init__()
        self.input_list = ['text', 'summary']
        self.name = "TextSummaryRNN"
        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)
        self.text_rnn = CellLayer(text_is_gru, self.embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.summary_rnn = CellLayer(summary_is_gru, self.embedding_layer.get_output_size(),
                                     summary_cell_hidden_size, summary_bidirectional, summary_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = summary_cell_hidden_size * 2 if summary_bidirectional else summary_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.summary_state = LastState(large_size2, large_size2)
        self.text_representation = SequentialModel([self.embedding_layer, self.text_rnn, self.text_state])
        self.summary_representation = SequentialModel([self.embedding_layer, self.summary_rnn, self.summary_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.summary_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        # self.seq = SequentialModel([self.concatenation, self.decision_layer])
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.summary_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class AttentionTextSummaryRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 trainable_embeddings=True,
                 text_is_gru=True,
                 text_cell_hidden_size=64,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 summary_is_gru=True,
                 summary_cell_hidden_size=32,
                 summary_stacked_layers=1,
                 summary_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 summary_att_mlp_layers=0,
                 summary_att_mlp_activation=relu,
                 summary_att_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):

        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param trainable_embeddings: boolean indicating if the embedding will be trainable or frozen
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the RNN for text representation
        :param text_bidirectional: boolean indicating if the RNN for text representation is bidirectional
        :param summary_is_gru: GRU cell type for the summary cell if true, otherwise LSTM
        :param summary_cell_hidden_size: the cell size of the RNN for summary representation
        :param summary_stacked_layers: the number of stacked layers of the RNN for summary representation
        :param summary_bidirectional: boolean indicating if the RNN for summary representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param summary_att_mlp_layers: number of layers of the summary attention mlp
        :param summary_att_mlp_activation: activation of the summary attention mlp
        :param summary_att_dropout: dropout of the summary attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(AttentionTextSummaryRNN, self).__init__()
        self.input_list = ['text', 'summary']
        self.name = "AttentionTextSummaryRNN"
        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=trainable_embeddings)

        self.text_rnn = CellLayer(text_is_gru, self.word_embedding_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.summary_rnn = CellLayer(summary_is_gru, self.word_embedding_layer.get_output_size(),
                                     summary_cell_hidden_size, summary_bidirectional, summary_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = summary_cell_hidden_size * 2 if summary_bidirectional else summary_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.summary_state = AttendedState(summary_att_mlp_layers, large_size2, summary_att_dropout, summary_att_mlp_activation)

        self.text_representation = SequentialModel([self.word_embedding_layer, self.text_rnn, self.text_state])
        self.summary_representation = SequentialModel([self.word_embedding_layer, self.summary_rnn, self.summary_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.summary_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.summary_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedTextSummaryRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=torch.tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 summary_is_gru=True,
                 summary_cell_hidden_size=32,
                 summary_stacked_layers=1,
                 summary_bidirectional=False,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param summary_is_gru: GRU cell type for the summary cell if true, otherwise LSTM
        :param summary_cell_hidden_size: the cell size of the RNN for summary representation
        :param summary_stacked_layers: the number of stacked layers of the RNN for summary representation
        :param summary_bidirectional: boolean indicating if the RNN for summary representation is bidirectional
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedTextSummaryRNN, self).__init__()
        self.input_list = ['text', 'summary']
        self.name = "ProjectedTextSummaryRNN"
        self.embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation
                                                     )
        self.text_rnn = CellLayer(text_is_gru, proj_hidden_size,
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.summary_rnn = CellLayer(summary_is_gru, proj_hidden_size,
                                     summary_cell_hidden_size, summary_bidirectional, summary_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = summary_cell_hidden_size * 2 if summary_bidirectional else summary_cell_hidden_size

        self.text_state = LastState(large_size1, large_size1)
        self.summary_state = LastState(large_size2, large_size2)

        self.text_representation = SequentialModel([self.embedding_layer,self.projection_layer, self.text_rnn, self.text_state])
        self.summary_representation = SequentialModel([self.embedding_layer, self.projection_layer, self.summary_rnn, self.summary_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.summary_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.summary_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output


class ProjectedAttentionTextSummaryRNN(Module):
    def __init__(self, embeddings,
                 embeddings_dropout=0.0,
                 proj_mlp_dropout=0.0,
                 proj_mlp_layers=1,
                 proj_mlp_activation=torch.tanh,
                 proj_hidden_size=100,
                 text_is_gru=True,
                 text_cell_hidden_size=128,
                 text_stacked_layers=1,
                 text_bidirectional=False,
                 summary_is_gru=True,
                 summary_cell_hidden_size=32,
                 summary_stacked_layers=1,
                 summary_bidirectional=False,
                 text_att_mlp_layers=0,
                 text_att_mlp_activation=relu,
                 text_att_dropout=0.0,
                 summary_att_mlp_layers=0,
                 summary_att_mlp_activation=relu,
                 summary_att_dropout=0.0,
                 top_mlp_layers=1,
                 top_mlp_activation=relu,
                 top_mlp_outer_activation=None,
                 top_mlp_dropout=0.0):
        """
        :param embeddings: the matrix of the embeddings
        :param embeddings_dropout: dropout of the embeddings layer
        :param proj_mlp_layers: number of layers of the projection mlp
        :param proj_mlp_activation: activation function of the projection mlp (usually tanh)
        :param proj_mlp_dropout: dropout of the projection
        :param proj_hidden_size: hidden size of the projection layers
        :param text_is_gru: GRU cell type for the text cell if true, otherwise LSTM
        :param text_cell_hidden_size: the cell size of the first RNN
        :param text_stacked_layers: the number of stacked layers of the first RNN
        :param text_bidirectional: boolean indicating if the first cell is bidirectional
        :param summary_is_gru: GRU cell type for the summary cell if true, otherwise LSTM
        :param summary_cell_hidden_size: the cell size of the RNN for summary representation
        :param summary_stacked_layers: the number of stacked layers of the RNN for summary representation
        :param summary_bidirectional: boolean indicating if the RNN for summary representation is bidirectional
        :param text_att_mlp_layers: number of layers of the text attention mlp
        :param text_att_mlp_activation: activation of the text attention mlp
        :param text_att_dropout: dropout of the text attention mlp
        :param summary_att_mlp_layers: number of layers of the summary attention mlp
        :param summary_att_mlp_activation: activation of the summary attention mlp
        :param summary_att_dropout: dropout of the summary attention mlp
        :param top_mlp_layers: number of layers for the top mlp
        :param top_mlp_activation: activation function of the top mlp layers - except the last layer
        :param top_mlp_outer_activation: activation function of the last layer of the top mlp (default None)
        :param top_mlp_dropout: dropout of the top mlp
        """
        super(ProjectedAttentionTextSummaryRNN, self).__init__()
        self.input_list = ['text', 'summary']
        self.name = "ProjectedAttentionTextSummaryRNN"

        self.word_embedding_layer = PretrainedEmbeddingLayer(embeddings, embeddings_dropout, trainable=False)
        self.projection_layer = MultiLayerPerceptron(num_of_layers=proj_mlp_layers,
                                                     dropout=proj_mlp_dropout,
                                                     init_size=self.word_embedding_layer.get_output_size(),
                                                     out_size=proj_hidden_size,
                                                     inner_activation=proj_mlp_activation,
                                                     outer_activation=proj_mlp_activation)



        self.text_rnn = CellLayer(text_is_gru, self.projection_layer.get_output_size(),
                                  text_cell_hidden_size, text_bidirectional, text_stacked_layers)
        self.summary_rnn = CellLayer(summary_is_gru, self.projection_layer.get_output_size(),
                                     summary_cell_hidden_size, summary_bidirectional, summary_stacked_layers)
        large_size1 = text_cell_hidden_size * 2 if text_bidirectional else text_cell_hidden_size
        large_size2 = summary_cell_hidden_size * 2 if summary_bidirectional else summary_cell_hidden_size

        self.text_state = AttendedState(text_att_mlp_layers, large_size1, text_att_dropout, text_att_mlp_activation)
        self.summary_state = AttendedState(summary_att_mlp_layers, large_size2, summary_att_dropout, summary_att_mlp_activation)
        self.text_representation = SequentialModel([self.word_embedding_layer, self.projection_layer, self.text_rnn, self.text_state])
        self.summary_representation = SequentialModel([self.word_embedding_layer, self.projection_layer, self.summary_rnn, self.summary_state])
        self.concatenation = ConcatenationLayer(self.text_representation, self.summary_representation)
        self.decision_layer = MultiLayerPerceptron(num_of_layers=top_mlp_layers,
                                                   init_size=large_size1 + large_size2,
                                                   out_size=1,
                                                   dropout=top_mlp_dropout,
                                                   inner_activation=top_mlp_activation,
                                                   outer_activation=top_mlp_outer_activation)
        self.params = list(filter(lambda p: p.requires_grad, self.parameters()))

    def forward(self, x, y):
        x = self.text_representation(x)
        y = self.summary_representation(y)
        output = self.concatenation(x, y)
        output = self.decision_layer(output)
        return output



