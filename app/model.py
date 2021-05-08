import os
root = os.path.dirname(os.path.abspath(__file__))
download_dir = os.path.join(root, 'nltk_data')

import nltk
nltk.data.path.append(download_dir)
