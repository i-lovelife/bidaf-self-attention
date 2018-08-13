virtualenv --python=python3.6 myenv && \
source myenv/bin/activate && \
pip install git+git://github.com/allenai/allennlp.git && \
pip install -r requirements.txt
