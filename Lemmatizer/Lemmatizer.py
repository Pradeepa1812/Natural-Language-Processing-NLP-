#stemmer===The word is not having enough meaning 
#-->Takes minimum to execute

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords  #stopwords basically means not mostly used words like the,is,of...

paragraph="""The Dr. A.P.J Abdul Kalam National Memorial was built in memory of Kalam
             by the DRDO in Pei Karumbu. In the island town of Rameswaram, Tamil Nadu
             It was inaugurated by Prime Minister Narendra Modi in July 2017.
             On display are the replicas of rockets and missiles which Kalam had worked with.
             Acrylic paintings about his life are also displayed along with hundreds
             of portraits depicting the life of the mass leader
             There is a statue of Kalam in the entrance showing him playing the Veena
             There are two other smaller statues of the leader in sitting and standing posture
             After graduating from the Madras Institute of Technology in 1960
             Kalam joined the Aeronautical Development Establishment of the Defence Research
             and Development Organisation by Press Information Bureau,
             Government of India as a scientist after becoming a member of the
             Defence Research & Development Service . He started his career by
             designing a small hovercraft, but remained unconvinced by his choice of
             a job at DRDO.[26] Kalam was also part of the INCOSPAR committee working 
             under Vikram Sarabhai, the renowned space scientist.
             In 1969, Kalam was transferred to the Indian Space Research Organisation 
             where he was the project director of India's first Satellite Launch Vehicle 
              which successfully deployed the Rohini satellite in near-earth orbit
             in July 1980; Kalam had first started work on an expandable rocket project independently at DRDO in 1965
             """
             
             
sentences=nltk.sent_tokenize(paragraph)
lemmatizer=WordNetLemmatizer()

#Lemmatizer
for i in range(len(sentences)):
       words=nltk.word_tokenize(sentences[i])
       words=[lemmatizer.lemmatize(word) for word in words if word not in set(stopwords.words('English'))]
       sentences[i]=' '.join(words)
             
