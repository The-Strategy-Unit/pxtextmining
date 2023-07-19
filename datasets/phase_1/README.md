Please note that the Care Opinion data is being shared under the [CC BY-NC-SA 4.0 licence](https://creativecommons.org/licenses/by-nc-sa/4.0/) and is generated from the [Care Opinion API](https://www.careopinion.org.uk/info/api-v2).

The dataset for phase 1 is stored in this folder. It is no longer used for training the pxtextmining models but is provided for historical interest.

The `co` and `co_multi_label` files are less useful, with fewer rows.

The main dataset is the file `text_data`. The following is a description of the columns:

code:
The shortcode given for the subcategory applied to the comment. There is 1:1 relationship between codes and subcategories, listed below.

 'cc': 'Care received',
 'xn': 'Nothing to improve',
 'sa': 'Attitude Of Staff',
 'ss': 'Staff: General',
 'cs': 'Advice and support',
 'mi': 'Amount/clarity of information',
 'sp': 'Professionalism/Competence Of Staff',
 'xe': 'Everything was good/bad',
 'mm': 'Communication',
 'cr': 'Rules/approach to care',
 'ml': 'Listening',
 'ef': 'Food',
 'wa': 'Time spent waiting for first appt/referral/service',
 'ap': 'Provision of services',
 'eq': 'Facilities/equipment',
 'ce': 'Emotional care',
 'ee': 'Environment/ facilities',
 'cp': 'Physical care',
 'aa': 'General',
 'ca': 'Activities',
 'co': '1-2-1 care/Time spent with service user',
 'cm': 'Medication ',
 'tc': 'Consistency/Continuity of care',
 'da': 'Respect For Diversity/ Person-Centeredness',
 'ec': 'Cleanliness',
 'sl': 'Staffing levels',
 'ti': 'Coordination/Integration Of Care',
 'cl': 'Made A Difference To My Life',
 'ds': 'Feeling safe including bullying',
 'tx': 'Transition And Discharge',
 'wb': 'Time spent waiting between appointments',
 'ct': 'Therapies',
 'al': 'Location',
 'dp': 'Involvement: Of Service Users/Patients',
 'dd': 'Dignity: General',
 'cf': 'Carer support',
 'xm': 'Miscellaneous',
 'tt': 'Transition/ coordination: General',
 'xg': 'Nothing was good',
 'ep': 'Parking/transport',
 'xf': 'Funding',
 'xl': 'Leave (under MHA)',
 'dc': 'Involvement: Of Family And Carers',
 'xs': 'Surveying'

label:
The overarching major category label for the text comment.

subcategory:
The subcategory label for the text comment.

feedback:
The actual text of the qualitative feedback comment.

criticality:
How critical the comment is towards the organisation. Can also be interpreted as a type of sentiment. Ranges from -5 to 5, with -5 being highly critical, or highly negative, and 5 being highly positive.

organization:
Which NHS Trust the feedback relates to.

question:
The question that the feedback relates to.

row_index:
row ID number for the feedback comment.
