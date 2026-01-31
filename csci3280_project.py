import pyaudio
import wave
import os
import whisper

def start_audio(time=10, save_file="test.wav", save_path = r"Directory"):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 16000
    RECORD_SECONDS = time
    WAVE_OUTPUT_FILENAME = os.path.join(save_path, save_file)  # Full path to save the file

    p = pyaudio.PyAudio()


    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording started")
    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording finished")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    return WAVE_OUTPUT_FILENAME  # Return the full path to the saved file

save_path = r"Directory"
save_file = "asr.wav"  # Changed to .wav extension

# Record audio and get the full path to the saved file
audio_file_path = start_audio(save_path=save_path, save_file=save_file)

# Load Whisper model
model = whisper.load_model("base")

# Transcribe the audio file using the full path
result = model.transcribe(audio_file_path)

output_text = result["text"]

# Save the transcription
output_path = os.path.join(save_path, "asr.txt")

print(f"User:{output_text}")

with open(output_path, "w", encoding="utf-8") as file:
    file.write(output_text)

print("Detected voice input successfully saved.")


import numpy as np
import matplotlib.pyplot as plt
import requests
import time
import re
from datetime import datetime
from matplotlib import gridspec
from datetime import timedelta
from openai import OpenAI
import threading
import queue
import time

#API_KEY = "XXXXXXXX"
DEEPSEEK_API_URL = "https://api.deepseek.com/summarize"
UNITS = "metric"
CITY_DICTIONARY = {
    "hong kong": "Hong Kong",
    "beijing": "Beijing",
    "shanghai": "Shanghai",
    "guangzhou":"Guangzhou",
    "shenzhen":"Shenzhen",
    "tianjin":"Tianjin",
    "tokyo": "Tokyo",
    "macao":"Macao",
    "taipei": "Taipei",
    "singapore": "Singapore",
    "bangkok":"Bangkok"
}


def read_forecast_days():
    try:
        with open('asr.txt', 'r') as file:
            content = file.read().strip()
            match = re.search(r'\d+', content)
            if match:
                days = int(match.group())
            else:
                raise ValueError("No numeric value found in asr.txt")

            return max(1, min(days, 5))

    except FileNotFoundError:
        print("Note: asr.txt not found, using default 5 days")
        return 5
    except Exception as e:
        print(f"Error reading forecast days: {e}, using default 5 days")
        return 5

def get_city_from_file():
    try:
        with open('asr.txt', 'r') as file:
            content = file.read().strip().lower()

            possible_cities = re.findall(r'\b[a-z]{2,}\b', content)

            for word in possible_cities:
                if word in CITY_DICTIONARY:
                    return CITY_DICTIONARY[word]

                for dict_name in CITY_DICTIONARY:
                    if word in dict_name.split():
                        return CITY_DICTIONARY[dict_name]

            print(f"No valid city found in asr.txt. Content: '{content}'")
            return None

    except FileNotFoundError:
        print("Error: asr.txt file not found")
        return None
    except Exception as e:
        print(f"Error reading city: {str(e)}")
        return None


def get_forecast(CITY, days):
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units={UNITS}"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        forecast_data = []
        for item in data['list']:
            local_time = datetime.strptime(item['dt_txt'], "%Y-%m-%d %H:%M:%S") + timedelta(hours=8)
            forecast_data.append({
                "datetime": local_time,
                "temp": item['main']['temp'],
                "humidity": item['main']['humidity'],
                "pressure": item['main']['pressure'],
                "weather": item['weather'][0]['main'],
                "description": item['weather'][0]['description'],
                "wind_speed": item['wind']['speed'],
                "wind_deg": item['wind'].get('deg', 0),
                "clouds": item['clouds']['all'],
                "pop": item.get('pop', 0),
                "rain_3h": item.get('rain', {}).get('3h', 0)
            })

        return forecast_data[:days * 8]

    except requests.exceptions.RequestException as e:
        print(f"Error fetching forecast data: {e}")
        return None

def analyze_daily_trends(forecast_data):
    daily_forecast = {}

    for item in forecast_data:
        date = item['datetime'].date()
        if date not in daily_forecast:
            daily_forecast[date] = {
                'temps': [],
                'humidities': [],
                'wind_speeds': [],
                'pop': []
            }

        daily_forecast[date]['temps'].append(item['temp'])
        daily_forecast[date]['humidities'].append(item['humidity'])
        daily_forecast[date]['wind_speeds'].append(item['wind_speed'])
        daily_forecast[date]['pop'].append(item['pop'])

    daily_analysis = []

    for date, data in daily_forecast.items():
        max_temp = max(data['temps'])
        min_temp = min(data['temps'])
        max_humidity = max(data['humidities'])
        min_humidity = min(data['humidities'])
        avg_wind_speed = sum(data['wind_speeds']) / len(data['wind_speeds'])
        max_pop = max(data['pop'])

        advice = []
        # Weather advice generation
        if max_temp > 33:
            advice.append("High temperature, stay hydrated and avoid direct sunlight.")
        elif min_temp < 12:
            advice.append("Low temperature, dress warmly.")

        if max_pop > 0.7:
            advice.append(f"High precipitation probability ({max_pop * 100:.0f}%), please carry an umbrella.")

        # Humidity advice
        if max_humidity > 80:
            advice.append("High humidity, it may feel uncomfortable outdoors.")
        elif min_humidity < 30:
            advice.append("Low humidity, stay hydrated and consider using moisturizer.")

        # Wind speed advice
        if avg_wind_speed > 10:
            advice.append(f"Strong winds ({avg_wind_speed:.1f} m/s), secure outdoor items.")

        daily_analysis.append({
            'date': date,
            'max_temp': max_temp,
            'min_temp': min_temp,
            'max_humidity': max_humidity,
            'min_humidity': min_humidity,
            'avg_wind_speed': avg_wind_speed,
            'max_pop': max_pop,
            'advice': advice if advice else ["Stable weather, no special alerts."]
        })

    return daily_analysis

def display_daily_forecast(daily_analysis):
    print("\n=== Daily Weather Forecast ===")
    for day in daily_analysis:
        print(f"\nDate: {day['date']}")
        print(f"Max Temperature: {day['max_temp']:.1f}°C")
        print(f"Min Temperature: {day['min_temp']:.1f}°C")
        print(f"Max Humidity: {day['max_humidity']:.0f}%")
        print(f"Min Humidity: {day['min_humidity']:.0f}%")
        print(f"Average Wind Speed: {day['avg_wind_speed']:.1f} m/s")
        print(f"Precipitation Probability: {day['max_pop'] * 100:.0f}%")
        print("Weather Advice:")
        for advice in day['advice']:
            print(f"- {advice}")

def plot_enhanced_forecast(forecast_data, analysis):
    fig = plt.figure(figsize=(15, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    dates = [x['datetime'] for x in forecast_data]

    ax1 = fig.add_subplot(gs[0, 0])
    temps = [x['temp'] for x in forecast_data]
    ax1.plot(dates, temps, 'r-o', label='Temperature')

    for i, date in enumerate(dates):
        ax1.annotate(f"{temps[i]:.1f}°C", xy=(date, temps[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)

    ax1.set_title('Temperature Trend')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Temperature (°C)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.legend()
    ax1.grid(True)

    ax2 = fig.add_subplot(gs[0, 1])
    humidity = [x['humidity'] for x in forecast_data]
    ax2.plot(dates, humidity, 'b-o', label='Humidity')

    for i, date in enumerate(dates):
        ax2.annotate(f"{humidity[i]:.0f}%", xy=(date, humidity[i]), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8)

    ax2.set_title('Humidity Trend')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Humidity (%)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.legend()
    ax2.grid(True)

    ax3 = fig.add_subplot(gs[1, 0], polar=True)
    wind_speeds = [x['wind_speed'] for x in forecast_data]
    wind_dirs = [np.radians(x['wind_deg']) for x in forecast_data]

    ax3.scatter(wind_dirs, wind_speeds, c=wind_speeds, cmap='viridis')
    ax3.set_title('Wind Speed/Direction', pad=20)
    ax3.set_theta_zero_location('N')
    ax3.set_theta_direction(-1)

    ax4 = fig.add_subplot(gs[1, 1])
    rain_prob = [x['pop'] * 100 for x in forecast_data]
    ax4.bar(dates, rain_prob, color='skyblue', alpha=0.7)
    ax4.set_title('Precipitation Probability')
    ax4.set_ylabel('Rain Probability (%)')
    ax4.set_ylim(0, 100)

    plt.tight_layout()
    plt.show()

def call_deepseek_api(user_message_content, output_queue):
    client = OpenAI(api_key="XXXXXXXXXX", base_url="https://api.deepseek.com/")
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a weather expert. \
                Please provide a vivid summary of the weather and offer tips to citizens. \
                    Output only plain text without any symbols or markdown (do not need to mention in your response). \
                        If encounter temperature symbol, please convert to Celcius."},
            {"role": "user", "content": user_message_content},
        ],
        stream=True
    )

    print("\nAI Response:")

    # LLM Generation in Streaming. When the end of a sentence is detected, it is output into the LLM-TTS Queue. It is also saved into forecast.txt.

    sentence = ""
    ai_response = []
    for event in response:
        if (not event.choices[0].finish_reason):
            if event.startswith(("!", ".", "?")):
                print(event.choices[0].delta.content, end='')
                sentence += event.choices[0].delta.content
                output_queue.put(sentence.strip())
                ai_response.append(sentence.strip())
                sentence = ""
            elif (event.startswith(",", "'", "%")):
                print(event.choices[0].delta.content, end='', flush=True)
                sentence += event.choices[0].delta.content
        else:
            print(" " + event.choices[0].delta.content, end='', flush=True)
            sentence += (" " + event.choices[0].delta.content)
        time.sleep(0.01)

    # End of LLM Generation
    output_queue.put(None)

    with open('forecast.txt', 'w', encoding='utf-8') as file:
        for line in ai_response:
            file.write(line + "\n")
    print("\nResponse saved to forecast.txt")

def main():
    CITY = get_city_from_file() or "Hong Kong"
    print(f"Fetching enhanced weather forecast for {CITY}...")

    days = read_forecast_days()
    forecast_data = get_forecast(CITY, days)
    user_message_content = f"Current weather data: {forecast_data}"

    if not forecast_data:
        print("Failed to get forecast data.")
        return

    daily_analysis = analyze_daily_trends(forecast_data)

    display_daily_forecast(daily_analysis)

    plot_enhanced_forecast(forecast_data, daily_analysis)

    answer = input("System: Do you want to call deepseek API? [y/n]: ")

    if answer.lower() == 'y':
        call_deepseek_api(user_message_content)


if __name__ == "__main__":
    main()


from google.cloud import texttospeech
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "sunlit-monolith-456716-f0-bb2f2944727d.json"

# This function converts text to speech using Google Cloud TTS API.

def google_tts(text: str, output_file: str = "output.mp3", speaking_rate: float = 0.75) -> None:
    """Convert text to speech using Google Cloud TTS API with adjustable speaking rate.

    Args:
        text: Input text to be synthesized
        output_file: Output filename (default: "output.mp3")
        speaking_rate: Speed of speech (0.25-4.0). 0.75 is slower than normal.
    """
    try:
        client = texttospeech.TextToSpeechClient()

        synthesis_input = texttospeech.SynthesisInput(text=text)

        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=speaking_rate
        )

        response = client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        with open(output_file, "wb") as out:
            out.write(response.audio_content)

        print(f"Audio saved to {output_file} (speaking rate: {speaking_rate})")

    except Exception as e:
        print(f"Error: {e}")

def read_forecast(file_path: str = "forecast.txt") -> str:
    """Read the content of forecast.txt file."""
    try:
        with open(file_path, "r") as file:
            content = file.read()
            print("Forecast content:")
            print(content)
            return content
    except FileNotFoundError:
        print(f"Error: File {file_path} not found!")
        return ""
    except Exception as e:
        print(f"Error reading file: {e}")
        return ""

if __name__ == "__main__":
    forecast_text = read_forecast()
    if forecast_text:
        google_tts(forecast_text, speaking_rate=0.75)