# MTG-ECPE

## Generative Joint Multi-task Learning for Emotion-Cause Pair Extraction

## Quick Links
- [Overview](#overview)
- [Code Usage](#code)
  - [Requirement](#requirement)
  - [Dataset](#data)
  - [LLMs](#llm)
  - [Run with Flan-T5](#runt5)
  - [Run with GPT-3.5](#GPT)
  - [Suggestions](#suggest)
- [MISC](#misc)


----------

## Overview<a name="overview" />

> The framework incorporates three parts: multi-task generative framework setting, multi-task learning objective, clause prediction normalization.

<p align="center">
  <img src="./figures/fig2.jpg" width="450"/>
</p>


\begin{table}[!h]
    \centering
    \scalebox{0.8}{
    \begin{tabular}{p{8.2cm}}
    \hline
    \underline{\centerline{ \textbf{Input}:}} 
         Task Descriptions:c1:The family currently relies on her for more than 2,000 yuan in wages to live, c2:the medical treatment has been heavily indebted, c3:if I disappeared, c4:and my wife did not have a Shenzhen household registration, c5:the days would be even more difficult, c6:it is helpless and heartbreaking, c7:but I also appreciate the netizens who help me, c8:and the family's unwavering commitment, c9:now I can only grit my teeth and continue to hold on, c10:and I hope to achieve the wish. \\ \hline
    \underline{\centerline{ \textbf{Ground Truth}:}  }
         Emotion Clauses: c7:but I also appreciate the netizens who help me \\
         Emotion Types: happiness \\
         ECPs: $\{c7,c7\},\{c7,c8\}$ \\  \hline
    \underline{\centerline{ \textbf{Target}: } }
         \textbf{EE Target}: [emotion clause: c7:but I also appreciate the netizens who help me, emotion type: happiness, keywords: appreciate] \\ 
         \hline
         \textbf{ECPE Target}: [emotion clause: c7:but I also appreciate the netizens who help me, cause clause: c7:but I also appreciate the netizens who help me, c8:and the family's unwavering commitment] \\ \hline
    \end{tabular}
    }
    \caption{Example of EE and ECPE tasks. The input is task description and all clauses. The target is the ECPs or emotion clauses.}
    % The input is the task description and a sequence of all clauses in the document, each annotated with its respective clause id. The output targets is the ECPs or emotion clauses.}
    \label{tab:illustrative example}
\end{table}


----------


----------

## Code Usage<a name="code" />


----------
### Requirement<a name="requirement" />

``` bash 
conda create -n thor python=3.8
```

``` bash
# CUDA 10.2
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=11.3 -c pytorch -c conda-forge
```

```bash
pip install -r requirements.txt
```

----------

### Dataset<a name="data" />

SemEval14 Laptop ([laptops](data%2Flaptops)) and Restaurant ([restaurants](data%2Frestaurants)), with fine-grained target-level annotations.



----------
### LLMs<a name="llm" />

A. Use the Flan-T5 as the backbone LLM reasoner:
  - [google/flan-t5-base](https://huggingface.co/google/flan-t5-base),  
  - [google/flan-t5-large](https://huggingface.co/google/flan-t5-large), 
  - [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl),  
  - [google/flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl),  

B. Evaluate with OpenAI [GPT-3.5](https://platform.openai.com/docs/models/gpt-3-5)

----------
### Training and Evaluating with Flan-T5<a name="runt5" />

Use the [main.py](main.py) script with command-line arguments to run the Flan-T5-based THOR system. 


```bash
python main.py -c <cuda_index> -r [thor|prompt] -d [restaurants|laptops] -z [True|False] -f <config_file>
```
Some important arguments:

- `-c`, `--cuda_index`: Index of the GPU to use for computation (default is 0).
- `-d`, `--data_name`: Name of the dataset. Choices are 'restaurants' or 'laptops' (default is 'laptops').
- `-r`, `--reasoning`: Specifies the reasoning mode, with one-step prompt or multi-step thor mode (default is 'thor').
- `-z`, `--zero_shot`: If True, the system directly performs zero-shot prediction, otherwise run the fine-tuning on the train set (default is True).
- `-f`, `--config`: Specifies the location of [config.yaml](config%2Fconfig.yaml) file.

Configurate more parameters in [config.yaml](config%2Fconfig.yaml) file.


----------

### Evaluating with GPT-3.5<a name="GPT" />

Go to the [eval_GPT](eval_GPT) fold, and run the [run_gpt_eval.py](eval_GPT%2Frun_gpt_eval.py) script:

```bash
python run_gpt_eval.py -k <openai_key> -d [restaurants|laptops]
```

Indicating your openai key. 
The reasoning traces and outputs of GPT for all instances are saved in `output_<data_name>.txt` file.

----------

### Suggestions<a name="suggest" />

- Suggest start with big enough LLM (e.g., `flan-t5-xl`), to better see the extraordinary reasoning ability.
- To tune the system with supervision, preferred with bigger batch size, and with large GPU ram; suggest with A100. 
- THOR is quite slower than the prompting mode.



----------

----------

## MISC<a name="misc" />

----------

### Citation

If you use this work, please kindly cite:

```
@inproceedings{FeiAcl23THOR,
  title={Reasoning Implicit Sentiment with Chain-of-Thought Prompting},
  author={Hao Fei, Bobo Li, Qian Liu, Lidong Bing, Fei Li, Tat-Seng Chua},
  booktitle = "Proceedings of the Annual Meeting of the Association for Computational Linguistics",
  pages = "1171--1182",
  year={2023}
}
```


----------


### Acknowledgement

This code is referred from following projects:
[CoT](https://arxiv.org/abs/2201.11903); 
[Flan-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5);
[OpenAI-GPT3](https://platform.openai.com/docs/models/gpt-3-5);
[Transformer](https://github.com/huggingface/transformers),


----------


### License

The code is released under Apache License 2.0 for Noncommercial use only. 



