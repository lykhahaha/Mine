import json

TRANSCRIPT_PATH = 'transcript-1-1.json'
SCRIPT_PATH = 'transcript-1-1.srt'
transcript = json.loads(open(TRANSCRIPT_PATH).read())
f = open(SCRIPT_PATH, 'w')


for counter, (time, content) in enumerate(transcript.items()):
    if counter == len(transcript) - 1:
        break
    text = f'{counter+1}\n'
    hour_1, hour_2 = int(time.split(':')[0]), int(list(transcript.keys())[counter+1].split(':')[0])
    time_text_1, time_text_2 = '', ''

    time_text_1 = f'01:{hour_1-60:02d}:' if hour_1 >= 60 else f'00:{hour_1:02d}:'
    time_text_2 = f'01:{hour_2-60:02d}:' if hour_2 >= 60 else f'00:{hour_2:02d}:'

    time_text_1 += f"{time.split(':')[1]},000"
    time_text_2 += f"{list(transcript.keys())[counter+1].split(':')[1]},000"
    time_text = time_text_1 + ' --> ' + time_text_2
    text += f'{time_text}\n{content}\n\n'
    f.write(text)
f.close()

# import youtube_dl
# ydl_opts = {}
# with youtube_dl.YoutubeDL(ydl_opts) as ydl:
#     ydl.download(['https://www.youtube.com/watch?v=XfoYk_Z5AkI&t=408s'])