# this file contains constant variables that are needed to build this project

import re
from extract_final_answer import extract_answer_gsm8k, extract_true_answer_aqua, extract_ai_answer_aqua, extract_true_answer_strategyqa, extract_ai_answer_strategyqa
from generate_fewshot_demonstrations import generate_singlecontext_fewshot_random_demonstrations_gsm8k, generate_singlecontext_fewshot_random_demonstrations_aqua, generate_singlecontext_fewshot_random_demonstrations_strategyqa

PATH_GSM8K_TRAIN = '../datasets/grade_school_math/data/train.jsonl'
PATH_GSM8K_TEST = '../datasets/grade_school_math/data/test.jsonl'
PATH_AQUA_TRAIN = '../datasets/AQuA/train.json'
PATH_AQUA_TEST = '../datasets/AQuA/test.json'
PATH_STRATEGYQA_TRAIN = '../datasets/Strategy_QA/strategyqa_train.json'
PATH_STRATEGYQA_DOCS = '../datasets/Strategy_QA/strategyqa_train_paragraphs.json'
ESTIMATE_COMPLETION_TOKENS = {'standard' : 3,
                              'cot' : 40}
COST_PER_TOKEN = {'gpt-3.5-turbo' : 0.002 / 1000}
INDEX_PATH = 'VectorStore/'
ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
PATTERN = r'[a-zA-Z]\)'
EXTRACT_ANSWERS_DIC = {'gsm8k' : 
                               {'extract_true_answer_func' : extract_answer_gsm8k,
                                'extract_ai_answer_func' : extract_answer_gsm8k},
                           'aqua' : 
                                {'extract_true_answer_func' : extract_true_answer_aqua,
                                 'extract_ai_answer_func' : extract_ai_answer_aqua},
                           'strategyqa' : 
                                {'extract_true_answer_func' : extract_true_answer_strategyqa,
                                 'extract_ai_answer_func' : extract_ai_answer_strategyqa}
                            }
PREFIX_DIC = {'gsm8k' : 
                          {'standard' : """You are willing to solve arithmetic math problems. The answer should not contain any special character. Follow the examples below and generate the answer using the format of these examples:""", 
                           'cot' : """You are willing to solve arithmetic math problems. Decompose the problem into intermediate steps and solve each step by generating the rationale. Explain the reasoning steps. Use the following format to answer the question: First generate intermediate reasoning steps, then generate the final answer as a single number. Here are some examples you can follow:\n\n"""
                          },
                  'aqua' : 
                          {'standard' : """"You are willing to solve algebraic word problems with multiple choice questions. Choose only one of the given options as the final answer. Follow the examples below and generate the answer using the format of these examples:\n\n""" ,
                           'cot' : """You are willing to solve algebraic word problems with multiple choice questions. First decompose the problem into intermediate reasoning steps, then solve and explain each intermediate step by generating the rationale. Then choose the final output to be only one of the given options. The output should include the rationale and the final output. Follow the examples below to generate the solution:\n\n"""
                          },
                  'strategyqa' : 
                          {'standard' : """You are willing to answer questions that require reasoning. The final answer must be either YES or NO. Follow the examples below and generate the answer using the format of these examples:""",
                           'cot' : """You are willing to answer questions that require reasoning. Decompose the problem into intermediate sub-questions to gather more information and generate a sub-answer to each sub-question before generating the final answer. The final answer must YES or NO. Follow the examples below and generate the answer using the format of these examples:\n\n"""
                          }
                  }

# This dictionary stores the functions which generate the context for each dataset
CONTEXT_FUNC_DIC = {'gsm8k' : generate_singlecontext_fewshot_random_demonstrations_gsm8k,
                   'aqua' : generate_singlecontext_fewshot_random_demonstrations_aqua,
                   'strategyqa' : generate_singlecontext_fewshot_random_demonstrations_strategyqa}

# This variables are used to build the suffix for various datasets and strategies
# For example, for GSM8K dataset, the suffix is:
suffix_gsm8k_standard = "Question: {question}\nOutput: "
rationale_answer_gsm8k = "\nRationale: \nOutput: "
suffix_gsm8k_cot = """\n\nQuestion: {question}""" + rationale_answer_gsm8k

# For AQUA dataset, the suffix is:
options_answer_aqua = "\nOptions: {}\nOutput: "
suffix_aqua_standard = "\n\nQuestion : {question}"
options_rationale_answer_aqua = "\nOptions: {}"
suffix_aqua_cot = "\n\nQuestion: {question}"

# For StrategyQA dataset, the suffix is:
suffix_strategyqa_cot = "Question: {question}\nOutput: "
suffix_strategyqa_standard = "Question: {question}\nOutput: "

# This dictionary contains the suffix for each dataset and strategy
SUFFIX_DIC = {'gsm8k' : {'standard': suffix_gsm8k_standard,
                        'cot' : suffix_gsm8k_cot
                    },
            'strategyqa' : {'standard' : suffix_strategyqa_standard,
                            'cot' : suffix_strategyqa_cot
                            },
            'aqua' : {'standard' : {'subset' : options_answer_aqua,
                                    'suffix' : suffix_aqua_standard},
                    'cot' : {'subset' : options_rationale_answer_aqua,
                                'suffix' : suffix_aqua_cot}}
            }

ZERO_SHOT_TEMPLATE_DIC = {'gsm8k': "You are willing to solve arithmetic math problems. Generate the final answer as a number in the end of the completion.\n Question: {question}\nLet's think step by step.",
                          'aqua' : '',
                          'strategyqa' : "You are willing to answer complex questions that require reasoning. The final answer must YES or NO.\nQuestion: {question}\nLet's think step by step."}
