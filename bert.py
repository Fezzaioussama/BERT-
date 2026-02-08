import torch
import torch.nn as nn


class BertEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, num_segments=2, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_len, embed_size)
        self.segment_embedding = nn.Embedding(num_segments, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.embed_size = embed_size

    def forward(self, token_ids, position_ids, segment_ids):
        token_embeddings = self.token_embedding(token_ids)
        position_embeddings = self.position_embedding(position_ids)
        segment_embeddings = self.segment_embedding(segment_ids)
        embeddings = token_embeddings + position_embeddings + segment_embeddings
        return self.dropout(embeddings)

class BertMultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.head_size = embed_size // num_heads
        self.qkv = nn.Linear(embed_size, embed_size * 3)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, x, mask=None):
        batch_size, seq_len, embed_size = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_size)
        qkv = qkv.permute(0, 2, 3, 1, 4)
        q, k, v = qkv.chunk(3, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_size ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        out = torch.matmul(attention, v)
        out = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, embed_size)
        return self.fc_out(out)

class BertTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.attention = BertMultiHeadAttention(embed_size, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)
        self.ffn = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        x = self.norm1(x + self.attention(x, mask))
        x = self.norm2(x + self.ffn(x))
        return x

class BertModel(nn.Module):
    def __init__(self, vocab_size, embed_size, max_len, num_heads, num_layers, num_segments=2, dropout=0.1):
        super().__init__()
        self.embedding = BertEmbedding(vocab_size, embed_size, max_len, num_segments, dropout)
        self.transformer_blocks = nn.ModuleList([BertTransformerBlock(embed_size, num_heads, dropout) for _ in range(num_layers)])

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        x = self.embedding(token_ids, position_ids, segment_ids)
        for block in self.transformer_blocks:
            x = block(x, mask)
        return x
            
class BertForMaskedLM(nn.Module):
    def __init__(self, bert_model, vocab_size):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(bert_model.embedding.embed_size, vocab_size)

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        outputs = self.bert(token_ids, position_ids, segment_ids, mask)
        return self.cls(outputs)

class BertForNextSentencePrediction(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(bert_model.embedding.embed_size, 2)

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        outputs = self.bert(token_ids, position_ids, segment_ids, mask)
        return self.cls(outputs[:, 0, :])

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(bert_model.embedding.embed_size, num_classes)

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        outputs = self.bert(token_ids, position_ids, segment_ids, mask)
        return self.cls(outputs[:, 0, :])

class BertForTokenClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.cls = nn.Linear(bert_model.embedding.embed_size, num_classes)

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        outputs = self.bert(token_ids, position_ids, segment_ids, mask)
        return self.cls(outputs)


class BertForQuestionAnswering(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model
        self.qa_outputs = nn.Linear(bert_model.embedding.embed_size, 2)

    def forward(self, token_ids, position_ids, segment_ids, mask=None):
        outputs = self.bert(token_ids, position_ids, segment_ids, mask)
        return self.qa_outputs(outputs)


if __name__ == "__main__":
    vocab_size = 30522
    embed_size = 768
    max_len = 512
    num_heads = 12
    num_layers = 12
    num_segments = 2
    dropout = 0.1
    bert_model = BertModel(vocab_size, embed_size, max_len, num_heads, num_layers, num_segments, dropout)
    print(bert_model)