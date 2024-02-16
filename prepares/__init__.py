# amazon
from .amazon.classifier import main as train_amazon_classifier
from .amazon.fluency import main as train_amazon_d_ppl
# caption
from .captions.classifier import main as train_caption_classifier
from .captions.fluency import main as train_caption_d_ppl
# gender
from .gender.classifier import main as train_gender_classifier
from .gender.fluency import main as train_gender_d_ppl
# political
from .political.classifier import main as train_political_classifier
from .political.fluency import main as train_political_d_ppl
# yelp
from .yelp.classifier import main as train_yelp_classifier
from .yelp.fluency import main as train_yelp_d_ppl
