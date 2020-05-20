FROM continuumio/miniconda3

# Grab requirements.txt.
ADD requirements.txt /tmp/requirements.txt

# Install dependencies
# RUN pip install -r /tmp/requirements.txt
# RUN conda install numpy
RUN python -m pip install -r /tmp/requirements.txt
RUN python -m pip install pynbody

# Add our code
ADD ./stream /opt/webapp/
WORKDIR /opt/webapp

# Python should be there: I use the requirements and then build pynbody...


RUN python -m pip install git+https://github.com/elehcim/simulation.git

#CMD gunicorn --bind 0.0.0.0:$PORT wsgi
CMD web: sh setup.sh && streamlit run stream.py