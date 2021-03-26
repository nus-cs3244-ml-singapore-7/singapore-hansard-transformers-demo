# Singapore Hansard 🤗 Transformers Demo
Demo using 🤗 Transformers (huggingface Transformers) on Singapore Hansard

## Named Entity Recognition

Uses pretrained
[`asahi417/tner-xlm-roberta-base-ontonotes5`](https://huggingface.co/asahi417/tner-xlm-roberta-base-ontonotes5)

Model is trained on OntoNotes Release 5.0 dataset and supports the Named Entities as specified in [OntoNotes Release 5.0 page 21](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf). Labels are in [IOB](https://en.wikipedia.org/wiki/Inside%E2%80%93outside%E2%80%93beginning_(tagging)) format.

```
O
B-person
I-person
B-group
I-group
B-facility
I-facility
B-organization
I-organization
B-geopolitical area
I-geopolitical area
B-location
I-location
B-product
I-product
B-event
I-event
B-work of art
I-work of art
B-law
I-law
B-language
I-language
B-date
I-date
B-time
I-time
B-percent
I-percent
B-money
I-money
B-quantity
I-quantity
B-ordinal number
I-ordinal number
B-cardinal number
I-cardinal number
```

### Samples

Mr Louis Ng Kok Kwang

```
M - O
r - O
Louis - B-person
Ng - I-person
Kok - I-person
K - I-person
wang - I-person
```

Ms Rahayu Mahzam

```
M - O
s - O
Raha - B-person
yu - I-person
Mah - I-person
zam - I-person
```

International Women's Day

```
International - B-event
Women - I-event
' - I-event
s - I-event
Day - I-event
```

the Licensing Terms and Conditions for AR Services (AR LTCs)

```
the - B-law
Li - I-law
cens - I-law
ing - I-law
Terms - I-law
and - I-law
Condi - I-law
tions - I-law
for - I-law
AR - I-law
Services - I-law
( - I-law

AR - B-law
L - I-law
TC - I-law
s - I-law
). - O

AR - B-law
L - I-law
TC - I-law
s - I-law
```

## Sentiment Analysis

Uses pretrained
[`textattack/bert-base-uncased-SST-2`](https://huggingface.co/textattack/bert-base-uncased-SST-2)

Expected negative sentiment

> Sir, I am proud that the Malay/Muslim community here remains strong and
> united despite many tests and challenges. It just that lately, some members of the community
> and our asatizah have complained that when crucial issues are voiced out or problems are
> raised, there are no clear answers from the authorities responsible for managing our welfare,
> needs and well-being.

```Negative: 0.8860296607017517, Positive: 0.11397039145231247```

---

Expected positive sentiment

> The difficult adjustments made by the community in response to COVID-19s
> was possible because our religious leadership was decisive and united. MUIS played a central
> role in this by issuing religious guidance early and rallying asatizah and mosque leaders to
> guide the community to adapt to the changing environment. While MUIS was monitoring
> developments and decisions of religious authorities around the world, we could not simply
> copy what others had done, but rather had to find our own solutions. When difficult decisions
> were made, the community came together and supported these decisions. Everyone played a part
> to encourage and guide one another – lawyers, doctors, community leaders and ordinary citizens,
> all rolled up our sleeves.

```Negative: 0.0032350728288292885, Positive: 0.9967648983001709```

---

Unclear example, negative topic but neutral sentiment

> One of the concerns I have relates to the guidance of high-risk Malay/Muslim
> youths who often come from families with many issues. They may have insufficient role models at
> home for a variety of reasons. It is not uncommon for such youths to suffer low confidence, poor
> performance in studies, or be involved in activities that may lead them down the road of
> juvenile delinquency.

```Negative: 0.983299970626831, Positive: 0.01669997349381447```

---

Unclear example, positive topic but possible negative sentiment

> Earlier this year, MCCY had organised "Emerging Stronger Together" and
> "Ciptasama" sessions for the Malay/Muslim community to share their views and suggestions
> in building a community of success and a better Singapore for the future. I note from the
> "Emerging Stronger" website that only 6% of the participants in these conversations were
> Malay. And I hope that more in our community can participate

```Negative: 0.042257506400346756, Positive: 0.9577425122261047```

---


## Summarisation

Uses pretrained
[`google/pegasus-cnn_dailymail`](https://huggingface.co/google/pegasus-cnn_dailymail)

### Input

> Mr Louis Ng Kok Kwang asked the Minister for Health
> whether couples who have pre-implantation genetically screened embryos stored overseas can have
> their embryos shipped to Singapore given current travel restrictions during the
> pandemic. The Parliamentary Secretary to the Minister for Health
> (Ms Rahayu Mahzam) (for the Minister for Health): Happy International Women's
> Day to all! During the pandemic, MOH received appeals from some couples to import their
> pre-implantation genetically screened embryos stored overseas. In reviewing
> each appeal, the Ministry considered whether processes and standards employed by overseas
> assisted reproduction (AR) centres are aligned to Singapore’s regulatory requirements under
> the Licensing Terms and Conditions for AR Services (AR LTCs). The Ministry may on an
> exceptional basis allow importation of the embryos, subject to conditions. These conditions
> include: (a) declaration by the overseas AR centre that the relevant requirements under the AR
> LTCs, including the handling, processing and storage of the embryos, are adhered to; (b) that
> no other findings besides the presence or absence of chromosomal aberrations are reported, and
> (c) proper documentation of the screening test results that were provided to the patient and
> attending physician in our local AR centres. Local AR centres which receive the
> tested embryos must also continue to ensure compliance with the AR LTCs.

### Output
> During the pandemic, MOH received appeals from some couples to import their pre-implantation
> genetically screened embryos stored overseas. The Ministry may on exceptional basis allow
> importation of the embryos, subject to conditions.
