from transformers.models.bert import modeling_bert

for symbol in dir(modeling_bert):
    if not symbol.startswith("_"):
        globals()[symbol] = getattr(modeling_bert, symbol)
