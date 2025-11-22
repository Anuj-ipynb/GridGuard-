from pyngrok import ngrok
import subprocess
import time

# Start your Streamlit app in background
print("Starting GridGuard Pro...")
subprocess.Popen(["streamlit", "run", "app.py", "--server.port=8501"])

# Wait 5 seconds for Streamlit to load
time.sleep(5)

# Create public ngrok tunnel
public_url = ngrok.connect(8501)
print("="*60)
print("GRIDGUARD PRO IS NOW LIVE WORLDWIDE!")
print("Share this link with your guide/examiner:")
print(public_url)
print("="*60)

# Keep the tunnel alive forever
ngrok_process = ngrok.get_ngrok_process()
ngrok_process.proc.wait()
input("Press Ctrl+C to stop the server...")