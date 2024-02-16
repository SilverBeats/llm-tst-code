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

API_KEYS = {
    # key must be same to LLMType.abbr
    'qwen': [
        'sk-d957c04216e545efaffaa15a9756f495',  # silverbeats: qwen
        'sk-18b5d398050241b9aa229a3267960174',  # gu: qwen
        'sk-3bbb324db19f4b8b86b5e0ee9363b7d8',  # xiao: qwen
        'sk-e32247edffbe4571b9ee0693df8df8fa',  # deng: qwen
        # 'sk-bfe1185a8ec64c3db541e2a5b0c56f35',  # huoqingao
        # 'sk-891c87baa0714ba58788b652a8e092e4'  # wanghua
    ],
    'gpt': [
        # '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',
        'sk-hXNOieAjL0m7PfP547AbEcBc0a204273B2962bFe14EaF8A2'
    ],
}

# key must be the same to LLMType.abbr
TEMPLATES = {
    'common': [
        'You are a linguist.'
        'You need to complete a text style transfer task.'
        'I will give you a {} style sentence, please change it to a {} style sentence.'
        'Please give me the revised sentence directly, without explaining the revision process. Sentence: {}',
        'Please use Python code to convert {} style sentence to {} style sentence.'
        'Please give me the revised sentence directly, without explaining the revision process. Sentence: {}',
    ],
    'special': {
        'yelp': [
            'You are a key opinion leader.'
            'You have a lot of experience reviewing stores and can give professional reviews.'
            'You need to complete a text style transfer task.'
            'I will give you a {} review of a store, please change it to a {} review.'
            'Please give me the revised sentence directly, without explaining the revision process. Sentence: {}'
        ],
        'amazon': [
            'You are an online reviewer with a lot of experience shopping online and reviewing products.'
            'You need to complete a text style transfer task.'
            'I will give you a {} review of a store, please change it to a {} review.'
            'Please give me the revised sentence directly, without explaining the revision process. Sentence: {}'
        ],
        'gender': [
            'You are a linguist who has studied the relationship between gender and speech for many years. '
            'You know the differences between male and female speaking styles.'
            'You need to complete a text style transfer task.'
            'I will give you a {} style sentence, please change it to a {} style sentence.'
            'Please give me the revised sentence directly, without explaining the revision process.Sentence: {}'
        ],
        'political': [
            'You are an American politician who has been in politics for many years.'
            'You are very familiar with the style of speech habits of the Republican Party and the Democratic Party.'
            'You need to complete a text style transfer task.'
            'I will give you a {} style sentence, please change it to a {} style sentence.'
            'Please give me the revised sentence directly, without explaining the revision process.Sentence: {}'
        ],
        'imagecaption': [
            'You are a humorous and romantic person.'
            'You know how to present something in a humorous or romantic way.'
            'You need to complete a text style transfer task.'
            'I will give you a {} style sentence, please change it to a {} style sentence.'
            'Please give me the revised sentence directly, without explaining the revision process.Sentence: {}'
        ]
    }
}
