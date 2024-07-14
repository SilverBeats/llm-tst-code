from enum import Enum, unique


@unique
class Task(Enum):
    R = 'rewrite'
    P = 'process'
    E = 'evaluate'

    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

@unique
class AccModelType(Enum):
    PLM = 'plm'
    FASTTEXT = 'fasttext'
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value

@unique
class FluencyModelType(Enum):
    PLM = 'plm'
    def __str__(self):
        return self.value

    def __repr__(self):
        return self.value


@unique
class LLMType(Enum):
    # serial num, llm name, abbreviation, use api ?
    llama2_7b_chat_hf = (1, 'llama2-7b-chat-hf', 'llama2', False)
    llama2_13b_chat_hf = (2, 'llama2-13b-chat-hf', 'llama2', False)
    qwen_7b_chat = (3, 'qwen-7b-chat', 'qwen', True)
    qwen_14b_chat = (4, 'qwen-14b-chat', 'qwen', True)
    gpt_3_5_turbo = (5, 'gpt-3.5-turbo', 'gpt', True)
    gpt_4 = (6, 'gpt-4', 'gpt', True)
    mistral_7b_instruct = (7, 'mistral-7b-instruct', 'mistral', False)

    @property
    def serial(self) -> int:
        return self.value[0]

    @property
    def type(self) -> str:
        return self.value[1]

    @property
    def abbr(self) -> str:
        return self.value[2]

    @property
    def useApi(self) -> bool:
        return self.value[3]


DATASET_TO_TST_TYPE = {
    'yelp': 'pos-neg',
    'amazon': 'pos-neg',
    'gender': 'male-female',
    'imagecaption': 'romantic-humorous',
    'political': 'republican-democratic'
}

ACCEPTABLE_DATASET = list(DATASET_TO_TST_TYPE.keys())

ACCEPTABLE_LLM = [item for item in LLMType]
