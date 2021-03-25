import torch
from transformers import BertTokenizer, BertForSequenceClassification

RAW_SEQUENCES = [
# Expected negative sentiment
"""Sir, I am proud that the Malay/Muslim community here remains strong and \
united despite many tests and challenges. It just that lately, some members of the community \
and our asatizah have complained that when crucial issues are voiced out or problems are \
raised, there are no clear answers from the authorities responsible for managing our welfare, \
needs and well-being.""",

# Expected positive sentiment
"""The difficult adjustments made by the community in response to COVID-19 \
was possible because our religious leadership was decisive and united. MUIS played a central \
role in this by issuing religious guidance early and rallying asatizah and mosque leaders to \
guide the community to adapt to the changing environment. While MUIS was monitoring \
developments and decisions of religious authorities around the world, we could not simply \
copy what others had done, but rather had to find our own solutions. When difficult decisions \
were made, the community came together and supported these decisions. Everyone played a part \
to encourage and guide one another – lawyers, doctors, community leaders and ordinary citizens, \
all rolled up our sleeves.""",

# Unclear example, negative topic but neutral sentiment
"""One of the concerns I have relates to the guidance of high-risk Malay/Muslim \
youths who often come from families with many issues. They may have insufficient role models at \
home for a variety of reasons. It is not uncommon for such youths to suffer low confidence, poor \
performance in studies, or be involved in activities that may lead them down the road of \
juvenile delinquency. """,

# Unclear example, positive topic but possible negative sentiment
"""Earlier this year, MCCY had organised "Emerging Stronger Together" and \
"Ciptasama" sessions for the Malay/Muslim community to share their views and suggestions \
in building a community of success and a better Singapore for the future. I note from the \
"Emerging Stronger" website that only 6% of the participants in these conversations were \
Malay. And I hope that more in our community can participate"""
]

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(sequences):
    # Pretrained model from https://huggingface.co/textattack/bert-base-uncased-SST-2
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'textattack/bert-base-uncased-SST-2').to(DEVICE)
    model.eval()

    inputs = tokenizer(sequences, return_tensors="pt", padding=True)
    input_ids = inputs["input_ids"].to(DEVICE)
    attention_mask = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask).logits
        predictions = torch.softmax(outputs, dim=-1)

    for sequence, prediction in zip(sequences, predictions):
        print(sequence)
        print("Negative: {}, Positive: {}".format(prediction[0], prediction[1]))
        print("--------")

if __name__ == '__main__':
    main(RAW_SEQUENCES)
