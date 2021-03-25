import torch
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification

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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(sequence):
    # Pretrained model from https://huggingface.co/asahi417/tner-xlm-roberta-base-ontonotes5
    tokenizer = XLMRobertaTokenizerFast.from_pretrained('xlm-roberta-base')
    model = XLMRobertaForTokenClassification.from_pretrained(
        'asahi417/tner-xlm-roberta-base-ontonotes5').to(DEVICE)
    model.eval()

    # Bit of a hack to get the tokens with the special tokens
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))
    inputs = tokenizer.encode(sequence, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model(inputs).logits
        predictions = torch.argmax(outputs, dim=2)[0]

    id2label = model.config.id2label
    print(id2label)

    for token, prediction in zip(tokens, predictions):
        print("{} - {}".format(token, id2label[prediction.item()]))

if __name__ == '__main__':
    main(RAW_SEQUENCE)
