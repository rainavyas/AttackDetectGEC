# Task
A Universal Concatenation adversarial attack is used to compromise a Grammatical Error Correction (GEC) system (code for attack is [here](https://github.com/rainavyas/ConcatAttackGEC)). The aim of this repository is to provide different detection-based defence schemes to be able to distinguish between real and adversarial samples. 

Here, experiments are performed on the [FCE public dataset](https://ilexir.co.uk/datasets/index.html) and a T5 based GEC system trained by _vennify_, available [here](https://huggingface.co/vennify/t5-base-grammar-correction?).

# Requirements

Clone this repository.

## Install with PyPI

`pip install happytransformer torch`
