import torch.nn.functional as F
from transformers import T5Tokenizer, T5Config
from models import monoT5

# model_name = 'unicamp-dl/mt5-base-mmarco-v2'
model_name = 'castorini/monot5-small-msmarco-100k'
config     = T5Config.from_pretrained('t5-small')
tokenizer  = T5Tokenizer.from_pretrained('t5-small')
model      = monoT5.from_pretrained(model_name, config=config)
model.set_tokenizer(tokenizer=tokenizer)
model.set_targets(['true', 'false'])

q = "Was there any political activism in Victor Jara's time?"
d0 = "From the years 1956 until 1959, Jazani had no political activities. His wife in this matter wrote: in these years \"We had no thoughts of a political future, we relied on our love and in the advancement of our studies. We spent most of our free time reading and at the theaters\". In 1959, he returned to politics and created a magazine named Nedaye Khalgh (\u0646\u062f\u0627\u06cc \u062e\u0644\u0642) with the goal of uniting the politic groups against the coup d'\u00e8tat regime. In the winter of 1959, because of the tight political climate the circulation of the magazine was halted."
d1 = "Victor Jara, pioneer of the Nueva Cancion Chilena movement and leftist political activist, remained true to his ideology right up to the moment of his death. He was tortured and murdered days after Salvador Allende, Chile\u2019s democratically elected president. On 16 September 1973, just a few days before his birthday, Victor Jara had his hands broken after already having suffered other forms of torture while being held captive in the  Chile Stadium ."

example_0 = f"Query: {q} Document: {d0} Relevant:"
example_1 = f"Query: {q} Document: {d1} Relevant:"

input_0 = tokenizer(example_0, return_tensors='pt')
input_1 = tokenizer(example_1, return_tensors='pt')

output_0 = model.predict(input_0)
output_1 = model.predict(input_1)

print(input_0.input_ids)
print(input_1.input_ids)
print(output_0)
print(output_1)

input = tokenizer([example_0, example_1], padding=True, return_tensors='pt')

output = model.predict(input)
print(input.input_ids)
print(output)

# [[0.5015087  0.49849123]
#  [0.9878755  0.01212445]]
