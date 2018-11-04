from models import ProjectedAttentionTextRNN, TextRNN, AttentionTextRNN, TextSummaryRNN, ProjectedTextRNN
from preprocess import generate_data
from utils import train
from torch.optim import Adam
from torch.nn import BCEWithLogitsLoss


def run_text():
    data = generate_data(emb_size=50, max_len=100)
    emb_matrix = data['emb_matrix']
    train_batches = data['train_batches']
    test_batches = data['test_batches']
    model = ProjectedAttentionTextRNN(emb_matrix, stacked_layers=1)
    optimizer = Adam(model.params, 0.001)
    criterion = BCEWithLogitsLoss()
    train(model, train_batches, test_batches, optimizer, criterion, 50, 5)


if __name__ == "__main__":
    run_text()
