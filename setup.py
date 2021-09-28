import setuptools

# Opens our README.md and assigns it to long_description.
with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="pxtextmining",
	version="0.3.4",
	author="Andreas D Soteriades",
	author_email="andreas.soteriades@nottshc.nhs.uk",
	description="Text Classification of Patient Experience feedback",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/CDU-data-science-team/pxtextmining",
	packages=setuptools.find_packages(),
	install_requires=[
    "blis==0.7.4", 
    "catalogue==1.0.0", 
    "certifi==2021.5.30", 
    "chardet==4.0.0", 
    "click==8.0.1", 
    "cycler==0.10.0", 
    "cymem==2.0.5", 
    "emojis==0.6.0", 
    "idna==2.10", 
    "imbalanced-learn==0.7.0", 
    "joblib==1.0.1", 
    "kiwisolver==1.3.1", 
    # "matplotlib>=3.3.2",
    "matplotlib",
    "murmurhash==1.0.5", 
    "mysql-connector-python==8.0.23", 
    "nltk==3.5", 
    "numpy>=1.20.2",
    "pandas==1.2.3", 
    "pickleshare==0.7.5", 
    "Pillow>=8.3.2", 
    "plac==1.1.3", 
    "preshed==3.0.5", 
    "protobuf==3.17.2", 
    "pyparsing==2.4.7", 
    "python-dateutil==2.8.1", 
    "pytz==2021.1", 
    "regex==2021.4.4", 
    "requests==2.25.1", 
    "scikit-learn>=0.23.2",
    "scipy==1.6.3",
    "seaborn==0.11.0", 
    "six==1.16.0", 
    "spacy==2.3.5", 
    "SQLAlchemy==1.3.23", 
    "srsly==1.0.5", 
    "textblob==0.15.3", 
    "thinc==7.4.5", 
    "threadpoolctl==2.1.0", 
    "tqdm==4.61.0", 
    "urllib3==1.26.5", 
    "vaderSentiment==3.3.2", 
    "wasabi==0.8.2"
    ],
# I don't think dependency_links is supported anymore (https://stackoverflow.com/questions/64482089/python-setup-py-dependency-as-url-to-tar-or-git)
#    dependency_links=[
#        "https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-2.3.1/en_core_web_lg-2.3.1.tar.gz", 
#        "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.3.1/en_core_web_sm-2.3.1.tar.gz"
#    ],
	classifiers=[
	  "Development Status :: 3 - Alpha",
		"Programming Language :: Python :: 3.8",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],
	python_requires='>=3.6',
	project_urls={
    'Documentation': 'https://cdu-data-science-team.github.io/pxtextmining/index.html',
    'Funding': 'https://www.england.nhs.uk/',
    'Source': 'https://github.com/CDU-data-science-team/pxtextmining',
    'Tracker': 'https://github.com/CDU-data-science-team/pxtextmining/issues',
    'Lay audience': 'https://involve.nottshc.nhs.uk/blog/new-nhs-england-funded-project-in-our-team-developing-text-mining-algorithms-for-patient-feedback-data/',
    'Blog post': 'https://cdu-data-science-team.github.io/team-blog/posts/2020-12-14-classification-of-patient-feedback/',
    'Team': 'https://cdu-data-science-team.github.io/team-blog/'
},
)
