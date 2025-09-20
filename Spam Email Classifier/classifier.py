# Traditional Programming vs ML: Spam Filter

# --- Traditional Rule-based ---
def rule_based_spam_filter(email):
    spam_words = ["win", "free", "lottery", "prize", "money"]
    return "spam" if any(word in email.lower() for word in spam_words) else "not spam"

print("Rule-based:", rule_based_spam_filter("You won a free lottery!"))  # spam
print("Rule-based:", rule_based_spam_filter("Meeting at 10 AM"))        # not spam


# --- Machine Learning Approach ---
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Step 1: Training Experience (dataset)

train_emails = [
    # --- Spam ---
    "Win a free lottery now!",
    "Claim your prize money today",
    "You won a cash reward",
    "Free money just for you",
    "Exclusive deal: get a free iPhone",
    "Congratulations! You have won a holiday trip",
    "Act now to claim your free gift card",
    "Get rich quick with this amazing offer",
    "Limited time offer: win cash prizes",
    "Earn money working from home easily",
    "Lowest price guaranteed, buy now",
    "Special promotion: 50% discount today only",
    "Click here to claim your free bonus",
    "This is not a scam! Win big rewards",
    "Final reminder: collect your free prize",

    # --- Not Spam ---
    "Project meeting at 10 AM",
    "Don't forget the assignment deadline",
    "Schedule your appointment tomorrow",
    "Join us for dinner tonight",
    "Looking forward to our weekend trip",
    "Can we reschedule our call to 3 PM?",
    "Your invoice for last month is attached",
    "Reminder: doctor appointment at 5 PM",
    "Team outing scheduled for next Friday",
    "Let’s catch up for coffee tomorrow",
    "Please review the attached report",
    "Don’t forget to bring your ID card",
    "Birthday party invitation for Saturday",
    "Your package will be delivered today",
    "See you at the conference next week"
]

# Labels (1 = spam, 0 = not spam)
train_labels = [
    # Spam
    1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
    # Not Spam
    0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
]

# Step 2 & 3: Convert text → numeric representation
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(train_emails)

# print("Vocabulary:", vectorizer.get_feature_names_out())
print("Vectorized Emails:\n", X.toarray())

# Step 4: Train algorithm (Naive Bayes)
model = MultinomialNB()
model.fit(X, train_labels)

# Step 5: Test final design
test_emails = [
    "Win a brand new car today",        # spam
    "Meeting rescheduled to 5 PM",      # not spam
    "Get your free vacation package",   # spam
    "Your homework submission is due",  # not spam
    "Congratulations, you won tickets"  # spam
]

X_test = vectorizer.transform(test_emails)
predictions = model.predict(X_test)

for email, pred in zip(test_emails, predictions):
    print(f"ML Model: '{email}' →", "spam" if pred == 1 else "not spam")