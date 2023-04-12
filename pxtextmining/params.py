
dataset = 'datasets/hidden/multilabel_merged230412.csv'

model_name = "distilbert-base-uncased"

minor_cats = ['Gratitude/ good experience',
              'Negative experience',
              'Not assigned',
              'Organisation & efficiency',
              'Funding & use of financial resources',
              'Collecting patients feedback',
              'Non-specific praise for staff',
              'Non-specific dissatisfaction with staff',
              'Staff manner & personal attributes',
               'Number & deployment of staff',
               'Staff responsiveness',
               'Staff continuity',
               'Competence & training',
               'Unspecified communication',
               'Staff listening, understanding & involving patients',
               'Information directly from staff during care',
               'Information provision & guidance',
               'Being kept informed, clarity & consistency of information',
               'Service involvement with family/ carers',
               'Patient contact with family/ carers',
               'Contacting services',
               'Appointment arrangements',
               'Appointment method',
               'Timeliness of care',
               'Supplying & understanding medication',
               'Pain management',
               'Diagnosis & triage',
               'Referals & continuity of care',
               'Length of stay/ duration of care',
               'Admission',
               'Discharge',
               'Care plans',
               'Patient records',
               'Impact of treatment/ care',
               'Links with non-NHS organisations',
               'Cleanliness, tidiness & infection control',
               'Sensory experience of environment',
               'Comfort of environment',
               'Atmosphere of ward/ environment',
               'Privacy',
               'Safety & security',
               'Provision of medical equipment',
               'Food & drink provision & facilities',
               'Service location',
               'Transport to/ from services',
               'Parking',
               'Activities & access to fresh air',
               'Electronic entertainment',
               'Feeling safe',
               'Patient appearance & grooming',
               'Equality, Diversity & Inclusion',
               'Mental Health Act',
               'Labelling not possible']


cat_map = {'Gratitude/ good experience': 'General',
              'Negative experience': 'General',
              'Not assigned': 'General',
              'Organisation & efficiency' : 'General',
              'Funding & use of financial resources': 'General',
              'Collecting patients feedback': 'General',
              'Non-specific praise for staff' : 'Staff',
              'Non-specific dissatisfaction with staff': 'Staff',
              'Staff manner & personal attributes': 'Staff',
               'Number & deployment of staff': 'Staff',
               'Staff responsiveness': 'Staff',
               'Staff continuity': 'Staff',
               'Competence & training': 'Staff',
               'Unspecified communication': "Communication & involvement",
               'Staff listening, understanding & involving patients': 'Communication & involvement',
               'Information directly from staff during care': "Communication & involvement",
               'Information provision & guidance': "Communication & involvement",
               'Being kept informed, clarity & consistency of information' :"Communication & involvement",
               'Service involvement with family/ carers': "Communication & involvement",
               'Patient contact with family/ carers': "Communication & involvement",
               'Contacting services': "Access to medical care & support",
               'Appointment arrangements': 'Access to medical care & support',
               'Appointment method': 'Access to medical care & support',
               'Timeliness of care' : 'Access to medical care & support',
               'Supplying & understanding medication': 'Medication',
               'Pain management': 'Medication',
               'Diagnosis & triage' : 'Patient journey & service coordination',
               'Referals & continuity of care': 'Patient journey & service coordination',
               'Length of stay/ duration of care': 'Patient journey & service coordination',
               'Admission': 'Patient journey & service coordination',
               'Discharge': 'Patient journey & service coordination',
               'Care plans': 'Patient journey & service coordination',
               'Patient records': 'Patient journey & service coordination',
               'Impact of treatment/ care': 'Patient journey & service coordination',
               'Links with non-NHS organisations' :'Patient journey & service coordination',
               'Cleanliness, tidiness & infection control': 'Environment & equipment',
               'Sensory experience of environment': 'Environment & equipment',
               'Comfort of environment': 'Environment & equipment',
               'Atmosphere of ward/ environment': 'Environment & equipment',
               'Privacy': 'Environment & equipment',
               'Safety & security': 'Environment & equipment',
               'Provision of medical equipment': 'Environment & equipment',
               'Food & drink provision & facilities': ' Food & diet',
               'Service location': 'Service location, travel & transport',
               'Transport to/ from services': 'Service location, travel & transport',
               'Parking': 'Service location, travel & transport',
               'Activities & access to fresh air': 'Activities',
               'Electronic entertainment': 'Activities',
               'Feeling safe': 'Category TBC',
               'Patient appearance & grooming' : 'Category TBC',
               'Equality, Diversity & Inclusion': 'Category TBC',
               'Mental Health Act': 'Mental Health specifics',
               'Labelling not possible': 'General'}

major_cats = set(cat_map.values())

merged_minor_cats = ['Gratitude/ good experience',
                    'Negative experience',
                    'Not assigned',
                    'Organisation & efficiency',
                    'Funding & use of financial resources',
                    'Collecting patients feedback',
                    'Non-specific praise for staff',
                    'Non-specific dissatisfaction with staff',
                    'Staff manner & personal attributes',
                    'Number & deployment of staff',
                    'Staff responsiveness',
                    'Staff continuity',
                    'Competence & training',
                    'Unspecified communication',
                    'Staff listening, understanding & involving patients',
                    'Information directly from staff during care',
                    'Information provision & guidance',
                    'Being kept informed, clarity & consistency of information',
                    'Contacting services',
                    'Appointment arrangements',
                    'Appointment method',
                    'Timeliness of care',
                    'Supplying & understanding medication',
                    'Pain management',
                    'Diagnosis & triage',
                    'Care plans',
                    'Impact of treatment/ care',
                    'Cleanliness, tidiness & infection control',
                    'Provision of medical equipment',
                    'Food & drink provision & facilities',
                    'Service location',
                    'Transport to/ from services',
                    'Parking',
                    'Activities & access to fresh air',
                    'Electronic entertainment',
                    'Mental Health Act',
                    'Labelling not possible',
                    'Dignity',
                    'Patient journey',
                    'Safety',
                    'Environment',
                    'Family/ carers']
