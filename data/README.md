---
license: cc-by-4.0
language:
- hi
- en
tags:
- code-mixing
- Hinglish
- expert-annotated
size_categories:
- 1M<n<10M
configs:  
  - config_name: LID
    data_files:
      - split: train
        path: "LID_train.csv"
      - split: test
        path: "LID_test.csv"
  - config_name: MLI
    data_files:     
      - split: train
        path: "MLI_train.csv"
      - split: test
        path: "MLI_test.csv"
  - config_name: POS
    data_files:
      - split: train
        path: "POS_train.csv"
      - split: test
        path: "POS_test.csv" 
  - config_name: NER
    data_files:
      - split: train
        path: "NER_train.csv"
      - split: test
        path: "NER_test.csv"
  - config_name: TN
    data_files:
      - split: train
        path: "TN_train.csv"
      - split: test
        path: "TN_test.csv"
  - config_name: MT
    data_files:
      - split: train
        path: "MT_train.csv"
      - split: test
        path: "MT_test.csv"
---

## Dataset Details

**COMI-LINGUA** (**CO**de-**MI**xing and **LING**uistic Insights on Natural Hinglish **U**sage and **A**nnotation) is a high-quality Hindi-English code-mixed dataset, manually annotated by three annotators. It serves as a benchmark for multilingual NLP models by covering multiple foundational tasks.

**COMI-LINGUA provides annotations for several key NLP tasks:**

* **Language Identification (LID):** Token-wise classification of Hindi, English, and other linguistic units.<br>
  Initial predictions were generated using the [Microsoft LID tool](https://github.com/microsoft/LID-tool), which annotators then reviewed and corrected.
```
  Example:
  sentence: प्रधानमंत्री  नरेन्द्र  मोदी  डिजिटल  इंडिया  मिशन  को  आगे  बढ़ाने  के  लिए  पिछले  सप्ताह  Google  के  CEO  सुंदर  पिचाई  से  मुलाकात  की  थी ।
  LID tags:   hi    hi   hi    en    en    en   hi  hi   hi  hi  hi   hi    hi     en   hi  en   hi   hi   hi   hi   hi  hi ot
```
* **Matrix Language Identification (MLI):** Sentence-level annotation of the dominant language.
```
  Example:
  sentence:"किसानों को अपनी फसल बेचने में दिक्कत न हो इसके लिये Electronic National Agriculture Market यानि ई-नाम योजना तेजी से काम हो रहा है।" Matrix Language: hi
  sentence:"India’s automation and design expert pool is vast, और ज़्यादातर Global companies के इंजीिनयिंरग center भी भारत में हैं।" Matrix Language: en
```
* **Part-of-Speech (POS) Tagging:** Syntactic categorization for linguistic analysis.<br>
  Tags were pre-assigned using the [CodeSwitch NLP library](https://github.com/sagorbrur/codeswitch), which annotators then reviewed and corrected.
```
  Example:
  sentence:  भारत   द्वारा  बनाया  गया Unified Payments Interface  यानि   UPI    भारत   की    एक   बहुत  बड़ी   success  story  है    ।
  POS tags: PROPN  ADP VERB VERB  PROPN   PROPN    PROPN    CONJ  PROPN  PROPN ADP  DET  ADJ  ADJ    NOUN    NOUN  VERB  X
```
* **Named Entity Recognition (NER):** Identification of entities in Hinglish text.<br>
  Each token is pre-assigned a language tag using the [CodeSwitch NLP library](https://github.com/sagorbrur/codeswitch), which annotators then reviewed and corrected.
```
  Example:
  sentence: "मालूम हो कि पेरिस स्थित Financial Action Task Force, FATF ने जून 2018 में पाकिस्तान को ग्रे लिस्ट में रखा था।"
  NER tags: "पेरिस": GPE, "Financial Action Task Force, FATF": ORGANISATION, "2018": Date, "पाकिस्तान": GPE   
```
* **Text Normalization (TN):** Sentence-level normalization of noisy, informal, and code-mixed Hinglish text to produce standardized outputs. <br>
  Common informal variants (e.g., hain / hai / hayn / hay / he → normalized to **hain** ) are identified and mapped to a consistent standard form.  
  Initial suggestions were generated using [GPT-OSS-120B](https://huggingface.co/openai/gpt-oss-120b), which annotators then reviewed and corrected.
```
  Example:  
  sentence: Janmdin ki dheron shubhkaamnaayen sir... Aap swasth rahen... Dirghayu ho... Yahi hamare poore pariwaar ki shubhkaamnaayen hain  
  Normalized: Janmadin ki dheron shubhkaamnayein sir. Aap swasth rahe. Dirghayu ho, yahi hamare pure parivaar ki shubhkaamnayein hain.
```  
* **Machine Translation (MT):** Parallel translation of sentences in Romanized Hindi and Devanagari Hindi and English languages.
  Initial translation predictions were generated using the [Llama 3.3 LLM](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct), which annotators then refined and corrected.
```
  Example:
  Hinglish Sentence: "भारत में भी green growth, climate resilient infrastructure और ग्रीन transition पर विशेष रूप से बल दिया जा रहा है।"
  English Translation: "In India too, special emphasis is being given to green growth, climate resilient infrastructure, and green transition."
  Romanized Hindi Translation: "Bharat mein bhi green growth, climate resilient infrastructure aur green transition par vishesh roop se bal diya ja raha hai."
  Devnagari Hindi Translation: "भारत में भी हरित विकास, जलवायु सहनशील आधारिक संरचना और हरित संक्रमण पर विशेष रूप से बल दिया जा रहा है।"
```
### Dataset Description

- **Curated by:** [Lingo Research Group at IIT Gandhinagar](https://lingo.iitgn.ac.in/)
- **Funded by:** ANRF
- **Language(s) (NLP):** Bilingual (Hindi [hi], English [en])
- **Licensed by:** cc-by-4.0

### Dataset Usage Disclaimer ‼️

<span style="color:red">This dataset is a research preview and is subject to ongoing iterative updates. As such, it provides only limited safety measures.</span>

## Contact US ✉️

[Lingo Research Group at IIT Gandhinagar, India](https://labs.iitgn.ac.in/lingo/) </br> 
Mail at: [lingo@iitgn.ac.in](lingo@iitgn.ac.in)

## Citation

If you use this dataset, please cite the following work:
```
@inproceedings{sheth-etal-2025-comi,
    title = "{COMI}-{LINGUA}: Expert Annotated Large-Scale Dataset for Multitask {NLP} in {H}indi-{E}nglish Code-Mixing",
    author = "Sheth, Rajvee  and
      Beniwal, Himanshu  and
      Singh, Mayank",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-emnlp.422/",
    pages = "7973--7992",
    ISBN = "979-8-89176-335-7",
}
```