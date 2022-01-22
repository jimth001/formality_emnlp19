# Harnessing Pre-Trained Neural Networks with Rules for Formality Style Transfer

## 1. model outputs

The outputs of our methods is under the "**gyafc_model_outputs**" directory. The "**em_out**" means the result for "**Entertainment&Music**". The "**fr_out**" means the result for "**Family&Relationships**".

"**formal.gpt.ori**" is the result of "**GPT-Orig**"

"**formal.gpt.rule**" is the result of "**GPT-Rule**"

"**formal.gpt.ori_rules.ens**" is the result of "**GPT-Ensemble**"

"**formal.gpt.cat_no_share.ori_rule**" is the result of "**GPT-CAT**"

"**formal.gpt.hie.ori_rule**" is the result of "**GPT-HA**".

"**formal.gpt.cat.domain_cmb.ori_rule**" is the result of "**GPT-CAT**" trained on domain combined data.

## 2. evaluation scripts

We released our evaluation scripts for "**Formality**", "**BLEU**" and "**PINC**". Scripts for evluation are under the "**evaluate**" directory. Run "**evaluate_em.py**" or "**evaluate_fr.py**" can calculate the metrics for the model outputs("gyafc_model_output" should be under the "evaluate" directory).

We didn't release our code for "**Meaning**" because we just use [BERT](https://github.com/google-research/bert) to fine-tune on STS.

**References are not released directly because you should first get access to GYAFC dataset. See more in [Section 3.1](#contact).**

## 3. model scripts

The code of our method is under "./gpt", "./utils" and "./preprocess".

### 3.1 training data<div id="contact"></div>

The training data includes original GYAFC dataset and the outputs of a simple rule based system. To obtain our training data, you should first get the access to [GYAFC dataset](https://github.com/raosudha89/GYAFC-corpus). Once you have gained the access to GYAFC dataset, please forward the acknowledgment to rmwangyl@qq.com, then we will provide access to our training data and other materials for evaluation.

### 3.2 run

Please download this repo directly, then put "training_data" under './' and "gyafc_model_outputs" under './evaluate/'. Run "main.py"(under './gpt/') to perform our methods. 

We suggest to use Pycharm to run this project.
