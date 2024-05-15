from transformers import T5TokenizerFast, BertTokenizer

t5_model="google/flan-t5-xl"
t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model)

# encoded = t5_tokenizer.encode("AC")
# print(encoded)
pepe = [71, 272, 205, 309, 262, 377, 350, 454, 27, 446, 480, 301, 283, 445, 411, 276, 1593, 391, 180, 332, 412, 584, 549, 4, 476, 1027, 5498, 7640, 2823, 11253, 5080, 9089, 21320] # FROM AA-GG
print(t5_tokenizer.decode(pepe))
#print(t5_tokenizer.encode('AG'))