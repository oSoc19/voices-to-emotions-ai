# Voices To Emotions AI
Repository containing all training scripts for the Voices To Emotions AI

# Functions
## Internal Functions
Storage Trigger (GCP): The Storage Trigger is a function that triggers automatically whenever a file is dropped in the calldata bucket on Google Cloud.

Mfcc (GCP) [POST]: An HTTP endpoint that generates mfcc data from an audio file, the audio file should be a url. This function takes in a `uri` query parameter with the uri of the audio fragment.

## Public Functions
/api/upload (GCP) [POST]: An HTTP endpoint that takes in an audio file and drops it into the storage bucket. This function takes in multipart form data with an `audio` entry, which is an audio file and a `user_id` entry which is the target user’s id.
