## RecallNet
Improving the recall of ConceptNet. Leveraging LLMs to clean 2 merged and automatically constructed Knowledge bases, while maintaining high precision.

The goal is to create a new dataset, RecallNet, from 2 dataset constructed automatically in the study of Julien Romero and Simon Razniewski: 
Mapping and Cleaning Open Commonsense Knowledge Bases with Generative Translation, Julien Romero & Simon Razniewski, 2023,
https://arxiv.org/abs/2306.12766

## Dataset
The two dataset used in this project are the followings that we constructed in the study mentionned above.
    - quasimodo_gent_lm_based_inv_top10.tsv
    - ascent_gent_lm_based_inv_top10.tsv

Both files contains two columns: 
    - "triple": (subject, predicate, object)
    - "frequency"

For more detail on the frequency columns, please refer to the article.

We also have the dataset ConceptNet:
    - conceptnet_csk_spor.csv
This dataset has been used to perform some quick evaluation during the testing phase.


## Preprocessing

['loading_dataset.py'](scripts/loading_dataset.py)

The first step is to preprocess the datasets that we will be using.
The first script ['loading_dataset.py'] clean both tsv files, by creating a csv files with 3 columns : subject, predicate, object ; and removing any incomplete line, and any useless spaces.
This script also gives 2 functions 'loading_quasimodo()' and 'loading_ascent()' (also 'load_conceptnet()') that will be used in other script to load the corresponding dataset.


['split_chunk.py'](scripts/split_chunk.py)

Once the csv dataset are created, it is important to chunk them into smaller pieces, this will allow you to save intermediate results throughout the process. Since the whole process is long, it's better to do it this way.
The chunking will be made predicate-wise so, in your output folder, for example the quasimodo one, the architecture will be as follows:

quasimodo/
|-->atlocation/
    |-->atlocation_1.csv
    |-->atlocation_2.csv
    |-->atlocation_3.csv
    ...
|-->capableof/
    ...
...

You can adjust the size of your chunk by changing the value of 'CHUNK_SIZE'.

['evaluate_data.py'](scripts/evaluate_data.py)
This script is used for quick evaluation and comparison between Quasimodo and ConceptNet to quantify how many triples from Quasimodo are already in ConceptNet

['cn_explore.py'](scripts/cn_explore.py)
Same as before, this script can be used to print a few sample of ConceptNet triple in order to manualy observe the type of data in ConceptNet.

## Core Steps of the Cleaning Process

Initally, the pipeline was implemented using LangChain and Ollama, you can find those scripts in the folders: 
    - ['langchain_pipeline/'](scripts/langchain_pipeline/)
    - ['ollama_pipeline/'](scripts/ollama_pipeline/)
and the other main scripts are the one that start with 'LG' in the ['scripts/'](scripts/) folder.

For computation efficiency and computation device availability, the entire pipeline have been implemented using vLLM.
The paragraphs below describe this pipeline.

# Overall scripts
This section quickly describe the scripts that reused over the different steps.
    - ['vLLM_pipeline/batching.py'](scripts/vLLM_pipeline/batching.py)
    - ['vLLM_pipeline/output.py'](scripts/vLLM_pipeline/output.py)
    - ['vLLM_pipeline/predicate_registery.json'](scripts/vLLM_pipeline/predicate_registery.json.py)
    - ['vLLM_pipeline/config.py'](scripts/vLLM_pipeline/config.py)


['batching.py'] Define 2 functions:
    - 'make_batches()': devide a list of triple into sublist in order to make small batch to send to the LLM. 
    - 'format_batches()': transform a batch of triple (a list) into a numeroted text using the format "0: (subject, predicate, object)". This is used to create the prompt sent to the LLM. 

['output.py'] Defines the Pydantic schemas for the LLM's structured outputs at each pipeline stage

['config.py'] Centralizes all pipeline settings (model path, batch size, sampling params, vLLM runtime limits, file paths) in one place, each overridable via environment variables, so the same code can be reconfigured for different runs without touching the source.
Make sure to specify the right model path for your use.

['predicate_registery.json'] Per-predicate reference data (definition, scope, a canonical example, and few-shot scoring examples with reasoning) used to build each prompt.


# Step 1: Scoring and Classifying the triple
The goal of this step is to score every triples from the dataset over 3 differents metrics from 0 to 5: 
    - Meaninfulness: Does this triple express a coherent and plausible commonsense relation?
    - Typicality: Is the stated relation generally true?
    - Saliency: Would humans spontaneously mention this when describing the subject?
To do so we use the Meta model: meta-llama/Llama-3.1-8B-Instruct, available on HuggingFace.

The main scripts and files that orchtestrate this step are:
    - ['vLLM_classify.py'](scripts/vLLM_classify.py)
    - ['vLLM_pipeline/classify.py'](scripts/vLLM_pipeline/classify.py)
    - ['vLLM_pipeline/prompt_classify.txt'](scripts/vLLM_pipeline/prompt_classify.txt)


['classify.py'] instantiates the vLLM model and handles all communication with it, builds a per-predicate system prompt, formats triples into chat messages, runs batched inference with structured JSON output enforced via a Pydantic schema, and parses the responses back into BatchEvaluation objects (returning None on malformed output).

['vLLM_classify.py'] Orchestrates the scoring run: for each requested predicate, finds its chunk files, skips ones already processed (resumable), runs inference on each remaining chunk via the vLLM_pipeline module, and writes out the scored CSV plus a log file (timing, score distribution, prompt used) per chunk — with per-chunk error handling so one failure doesn't stop the whole run.

['vLLM_pipeline/prompt_classify.txt'] is the main prompt template for the classifying task that is completed by the previous script and given to the model.

To run the script, please make sure to use the following configuration :

uv run vLLM_classify_2.py \
    --data-dir quasimodo_chunked \ ## <-- adapt to your case, folder in which your chunked csv files are 
    --dataset-prefix q \ ## <-- specify q for Quasimodo and a for Ascent
    --predicates "at location" "causes" ...
    --fp-num 0

RESULT ARCHITECTURE:
One CSV + one TXT per processed chunk, grouped under a per-predicate subfolder. The prefix (q/a) comes from --dataset-prefix, fp_num identifies the run.
The CSV file contains the triple with their score and the TXT one contains the configuration of the run.

# Step 2: Splitting the results
Split the triple into 3 classes: KEEP, INBETWEEN, REJECT

The script ['hand_rule_splitting.py'](scripts/rule_mining/hand_rule_splitting.py) perform the splitting form a specified decision rule and save the triple into 2 two disctinct folders: "KEEP" and "INBETWEEN"

The user script in ['rule_mining/'](scripts/rule_mining/) have been used to evaluate the data and determine the final decision rule

# Step 3: Filtering the INBETWEEN folder
The goal of this step is to filter the triples in the INBETWEEN folder.
To do we ask a bigger LLM model to classify each triples into 3 classes :
    - KEEP
    - MODIFY
    - REJECT

The model used is the: meta-llama/Llama-3.3-70B-Instruct, available on HuggingFace.

The main scripts and files that orchtestrate this step are:
    - ['vLLM_filtering.py'](scripts/vLLM_filtering.py)
    - ['vLLM_pipeline/filtering.py'](scripts/vLLM_pipeline/filtering.py)
    - ['vLLM_pipeline/prompt_filtering.txt'](scripts/vLLM_pipeline/prompt_filtering.txt)

The script architecture on this step is the same as in 'Step 1'.
Run the main script as follows:

------
export RECALLNET_VLLM_MODEL="/mnt/ssd/recallnet/models/Meta-Llama-3.3-70B-Instruct"
export RECALLNET_MODEL_LIGHT="llama3.3:70b-fp8"
export RECALLNET_VLLM_MAX_LEN=4096
export RECALLNET_PROMPT="prompt_filter.txt"

uv run vLLM_filtering.py \
    --dataset-prefix q \
    --predicates "at location" ...
------

After this step, you may use the script ['evaluate_filtering.py'](scripts/evaluate_filtering.py) to organise your filtered data into 3 differents folder.

# Step 4: Modifying the triple
The goal of this step is to provide a modify of the triple classified as MODIFY in the previous step.
To do so, we use the same model, give the triple and first ask to re-evaluate the triple into KEEP, MODIFY and REJECT. This ensure another validation for some triple that may have been missed preivously.
If the verdict is MODIFY, the model has to provide a modification.

The main scripts and files that orchtestrate this step are:
    - ['vLLM_modifying.py'](scripts/vLLM_modifying.py)
    - ['vLLM_pipeline/modifying.py'](scripts/vLLM_pipeline/modifying.py)
    - ['vLLM_pipeline/prompt_modifying.txt'](scripts/vLLM_pipeline/modifying.txt)

The script architecture on this step is the same as in 'Step 1'.
Run the main script as follows:

------
export RECALLNET_VLLM_MODEL="/mnt/ssd/recallnet/models/Meta-Llama-3.3-70B-Instruct"
export RECALLNET_MODEL_LIGHT="llama3.3:70b-fp8"
export RECALLNET_VLLM_MAX_LEN=4096
export RECALLNET_PROMPT="prompt_modify.txt"

uv run vLLM_modifying.py \
    --dataset-prefix a \
    --predicates "at location" ...
------

After this step, you may use the script ['evaluate_modifying.py'](scripts/evaluate_filtering.py) to organise your filtered data into 3 differents folder.

# Step 5: Scoring the modified triple
This step does the same process as in the Step 1, but for the newly modified triple

The main scripts and files that orchtestrate this step are:
    - ['vLLM_classify_mod.py'](scripts/vLLM_classify_mod.py)
    - ['vLLM_pipeline/classify.py'](scripts/vLLM_pipeline/classify.py)
    - ['vLLM_pipeline/prompt_classify.txt'](scripts/vLLM_pipeline/prompt_classify.txt)

# Step 6: Splitting the rescored triple
This step does the same process as in the Step 2, but for the newly modified and scored triple.

# Step 7: Final filtering
This step does the same process as in the Step 3, but the triple labelled as INBETWEEN during the Step 6.
The difference now is that we only keep the triple labelled as KEEP from the 70B model.

The main scripts and files that orchtestrate this step are:
    - ['vLLM_filtering_mod.py'](scripts/vLLM_filtering_mod.py)
    - ['vLLM_pipeline/filtering.py'](scripts/vLLM_pipeline/filtering.py)
    - ['vLLM_pipeline/prompt_filtering.txt'](scripts/vLLM_pipeline/prompt_filtering.txt)
