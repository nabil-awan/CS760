####################################### Q 3.1 ####################################################
import tarfile
import os
tar_path = 'languageID.tar'
unpack_dir = 'languageID'
with tarfile.open(tar_path) as file:
    file.extractall(path=unpack_dir)
os.listdir(unpack_dir)[:10]  
unpack_dir = os.path.join(unpack_dir, 'languageID')
training_files = [file for file in os.listdir(unpack_dir) if file.endswith('.txt') and file[1] in '0123456789']

# Sorting the files by language and then by number
training_files.sort(key=lambda x: (x[0], int(x[1])))

from collections import defaultdict
def preprocess_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read().lower()  
        return ''.join(char for char in text if char.isalpha() or char.isspace())  

language_data = defaultdict(str)

# Processing each training file and aggregating the text by language
for file in training_files:
    if file[1] in '0123456789':  
        language = file[0]  
        file_path = os.path.join(unpack_dir, file)
        language_data[language] += preprocess_text(file_path)

num_documents = len(training_files) / 3  
num_classes = 3  
alpha = 0.5  

prior_probabilities = {}

for language in language_data.keys():
    prior_probabilities[language] = (num_documents + alpha) / (num_documents * num_classes + alpha * num_classes)

# Calculating the log-probabilities for the priors
log_prior_probabilities = {language: math.log(prob) for language, prob in prior_probabilities.items()}

# Displaying the log-prior probabilities
log_prior_probabilities

###################################### Q 3.2 #############################################
# Defining the vocabulary (26 letters and space)
vocabulary = set('abcdefghijklmnopqrstuvwxyz ')
vocab_size = len(vocabulary)
log_conditional_probabilities = defaultdict(lambda: defaultdict(float))

for language, text in language_data.items():
    total_count = len(text)  # Total count of characters in documents of the language
    
    char_counts = defaultdict(int)
    for char in text:
        if char in vocabulary:  # Considering only characters in the defined vocabulary
            char_counts[char] += 1
    
    for char in vocabulary:
        prob = (char_counts[char] + 0.5) / (total_count + 0.5 * vocab_size)  # Applying additive smoothing
        log_conditional_probabilities[language][char] = math.log(prob)

{language: {char: log_conditional_probabilities[language][char] for char in 'abc '} 
 for language in log_conditional_probabilities.keys()}

log_conditional_prob_english = log_conditional_probabilities['e']

# Converting the log-probabilities back to regular probabilities for presentation
conditional_prob_english = {char: math.exp(log_prob) for char, log_prob in log_conditional_prob_english.items()}
ordered_chars = sorted(conditional_prob_english.keys(), key=lambda c: (c == ' ', c))
theta_e = [conditional_prob_english[char] for char in ordered_chars]
theta_e
######################################## Q 3.3 ##########################################

def extract_conditional_probabilities(language):
    log_probs = log_conditional_probabilities[language]
    return [math.exp(log_probs[char]) for char in ordered_chars]

# Extracting the class conditional probabilities for Japanese and Spanish
theta_j = extract_conditional_probabilities('j')
theta_s = extract_conditional_probabilities('s')
theta_j, theta_s

####################################### Q 3.4 ##########################################

test_file_path = os.path.join(unpack_dir, 'e10.txt')
test_text = preprocess_text(test_file_path)
test_char_counts = defaultdict(int)
for char in test_text:
    if char in vocabulary:
        test_char_counts[char] += 1

x_vector = [test_char_counts[char] for char in ordered_chars]
x_vector

###################################### Q 3.5 ############################################

def calculate_log_probability_x_given_y(x_vector, log_conditional_prob_language):
    return sum(x_i * log_conditional_prob_language[char] for x_i, char in zip(x_vector, ordered_chars))

# Calculating the log-probabilities for each language
log_prob_x_given_e = calculate_log_probability_x_given_y(x_vector, log_conditional_probabilities['e'])
log_prob_x_given_j = calculate_log_probability_x_given_y(x_vector, log_conditional_probabilities['j'])
log_prob_x_given_s = calculate_log_probability_x_given_y(x_vector, log_conditional_probabilities['s'])

# Exponentiating the log-probabilities to get the actual probabilities
prob_x_given_e = math.exp(log_prob_x_given_e)
prob_x_given_j = math.exp(log_prob_x_given_j)
prob_x_given_s = math.exp(log_prob_x_given_s)

# Displaying the log-probabilities log(p(x|y)) for each language
log_prob_x_given_e, log_prob_x_given_j, log_prob_x_given_s

##################################### Q 3.6 ##########################################

log_posterior_e = log_prob_x_given_e + log_prior_probabilities['e']
log_posterior_j = log_prob_x_given_j + log_prior_probabilities['j']
log_posterior_s = log_prob_x_given_s + log_prior_probabilities['s']

# Identifying the language that maximizes the log-posterior probability
languages = ['e', 'j', 's']
log_posteriors = [log_posterior_e, log_posterior_j, log_posterior_s]
predicted_language = languages[log_posteriors.index(max(log_posteriors))]

# Displaying the log-posterior probabilities and the predicted language
log_posterior_e, log_posterior_j, log_posterior_s, predicted_language

##################################### Q 3.7 ##########################################

from collections import Counter
def classify_document(file_path):
    text = preprocess_text(file_path)
    char_counts = Counter(char for char in text if char in vocabulary)
    x_vector = [char_counts[char] for char in ordered_chars]
    
    
    log_posteriors = [calculate_log_probability_x_given_y(x_vector, log_conditional_probabilities[language])
                      + log_prior_probabilities[language] for language in languages]
    
    
    return languages[log_posteriors.index(max(log_posteriors))]

test_files = [file for file in os.listdir(unpack_dir) if file.endswith('.txt') and file[1] in '10111213141516171819']

confusion_matrix = {language: defaultdict(int) for language in languages}

for file in test_files:
    true_language = file[0]
    file_path = os.path.join(unpack_dir, file)
    predicted_language = classify_document(file_path)
    confusion_matrix[predicted_language][true_language] += 1

confusion_matrix



