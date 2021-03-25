import torch
from transformers import BertTokenizerFast, BertForTokenClassification

RAW_SEQUENCE = """<p>1 <strong>Mr Louis Ng Kok Kwang</strong> asked&nbsp;the Minister for Health 
whether couples who have pre-implantation genetically screened embryos stored overseas can have 
their embryos shipped to Singapore given current travel restrictions during the 
pandemic.</p><p><strong>	The Parliamentary Secretary to the Minister for Health 
(Ms Rahayu Mahzam) (for the Minister for Health)</strong>: Happy International Women's 
Day to all! During the pandemic, MOH received appeals from some couples to import their 
pre-implantation genetically screened embryos stored overseas.&nbsp;</p><p>In reviewing 
each appeal, the Ministry considered whether processes and standards employed by overseas 
assisted reproduction (AR) centres are aligned to Singapore’s regulatory requirements under 
the Licensing Terms and Conditions for AR Services (AR LTCs). The Ministry may on an 
exceptional basis allow importation of the embryos, subject to conditions. These conditions 
include: (a) declaration by the overseas AR centre that the relevant requirements under the AR 
LTCs, including the handling, processing and storage of the embryos, are adhered to; (b) that 
no other findings besides the presence or absence of chromosomal aberrations are reported, and 
(c) proper documentation of the screening test results that were provided to the patient and 
attending physician in our local AR centres.&nbsp;</p><p>Local AR centres which receive the 
tested embryos must also continue to ensure compliance with the AR LTCs.</p>"""

def main(sequence):
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
    model = BertForTokenClassification.from_pretrained('dbmdz/bert-large-cased-finetuned-conll03-english')
    
    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="pt")

    outputs = model(inputs).logits
    predictions = torch.argmax(outputs, dim=2)

    label_list = [
        "O",       # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",   # Beginning of a person's name right after another person's name
        "I-PER",   # Person's name
        "B-ORG",   # Beginning of an organisation right after another organisation
        "I-ORG",   # Organisation
        "B-LOC",   # Beginning of a location right after another location
        "I-LOC"    # Location
    ]

    for token, prediction in zip(tokens, predictions[0].numpy()):
        print("{} - {}".format(token, label_list[prediction.item()]))

if __name__ == '__main__':
    main(RAW_SEQUENCE)
