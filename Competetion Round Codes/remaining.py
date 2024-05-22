import requests

api_url = "http://13.53.169.72:5000/attempts/student"
team_id = "18u44K7"

res = requests.post(
    url=api_url,
    # headers={"Content-Type": "application/json"},
    json={"teamId": "18u44K7"},
)

print(res.status_code)
print(res.json())
