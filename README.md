## Insult Detector 0.0.1
The purpose of this Python application is to train an AI model to detect when what you hear is not from your ear drums.
Victims report hearing insults, threats and malicious disinformation. Insults are reported most commonly, hence this application is called insult detector. It is more precisely a clandestine communication detector. 

The application is currently experimental and needs testing.

# Hardware requirements
Purchase a cheap unidirectional BCI device for experiments at home. 
One option is Muse 2 which connects to your computer via Bluetooth.
It costs $250. 

Official product website: https://choosemuse.com/products/muse-2

Amazon: https://www.amazon.com/MUSE-Meditation-Biofeedback-Responsive-Neurofeedback/dp/B07HL2S9JQ/

You can use any computer that runs Python 3.

# Applied research
In applied research, you start with a problem that needs solving. You do a literature review and study previous solutions to the problem. Then, you should synthesize a new solution from the existing ones, and it should involve extending them in a meaningful way. 

# User stories
User story 1: A user will record his own EEG data from his Muse 2 or other BCI device, i.e. 10 seconds.

User story 2: A user will then label his EEG recording sample with the time when he heard an insult start and with its end time and save it as json.

User story 3: A user will train an AI model for feature detection to detect insults automatically based on his labeled data set. (Multiple labeled examples are required to make the AI model accurate. For example, the user will select 10, or 100 labeled recordings that are saved as .json).

User story 4: A user will process his EEG data in real-time and the application will display a warning using the trained AI model when any insulting starts and hide the warning when it ends.

# Additional resources for researchers
If you want to become qualified in the underlying science that explains how BCI and EEG work, read https://www.shorturl.at/hA0pe
