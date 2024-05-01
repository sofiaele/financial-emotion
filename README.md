# financial-emotion
Pridicting bank stock returns from emotion of central banker press conferences

STEPS:

1. Use csv with youtube links to download the audio files from youtube videos:
   - Run utils/preprocessing/download_audios_from_csv(csv_file)
   - Outcomes: .csv file with new column of audio file paths, audio files in `data/audio` folder
2. Run utils/preprocessing/create_csv(videos, returns)
   - Outcomes: introductory.csv, non_introductory.csv with merged youtube info in returns.
3. Run utils/preprocessing/mean_per_month(csv_file)
   - Outcomes: non_introductory_mean.csv with mean returns per month
4. (Optional): Merge previous csv with new one (non_introductory_mean.csv + final.csv)
   - Run utils/preprocessing/merge_csvs(csv1, csv2)
   - Outcomes: merged.csv
- The above mentioned csv serves as the final ground truth file.
5. python3 diarization-asr/whisper.py --audios path_to_audio_files
   - Outcomes: created csvs for each audio file with speaker diarization and ASR in the folder of audios
6. python3 diarization-asr/dominant_speaker.py --csvs path_to_csvs
   - Outcomes: csv with dominant speaker per audio file
7. python3 diarization-asr/split_audio_segments.py
   - Creates audio utterances and updates the csv with the path to the utterances
8. Send to behavioral models using client.py and utterances folder
9. Convert json results to csv 
   - python3 utils/api_json_to_csv.py --json path_to_json