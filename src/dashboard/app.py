from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def dashboard():
    # Placeholder data
    user_data = {
        'name': 'Alex',
        'email': 'alex.doe@email.com'
    }

    wellness_data = {
        'wellness_score': 82,
        'wellness_score_change': 3,
        'avg_mood': 7.8,
        'avg_mood_change': 0.4,
        'checkins_completed': '5/7',
        'checkins_completed_change': 1,
        'mood_trends': {
            'value': '7.2/10',
            'change': '+0.5%'
        },
        'stress_trends': {
            'value': '4.5/10',
            'change': '-1.2%'
        }
    }

    recommendations = [
        {
            'icon': 'self_improvement',
            'title': 'Try a 5-min Meditation',
            'description': 'A short breathing exercise can help lower your stress.'
        },
        {
            'icon': 'forum',
            'title': 'Join a Peer Support Group',
            'description': 'Connect with others who share similar experiences.'
        },
        {
            'icon': 'menu_book',
            'title': 'Read about Cognitive Reframing',
            'description': 'Learn a new technique to manage negative thoughts.'
        }
    ]

    goals = [
        {
            'progress': '2/3',
            'description': 'Meditate 3x a week',
            'stroke_offset': 33
        },
        {
            'progress': '9/10',
            'description': 'Sleep 7+ hours',
            'stroke_offset': 10
        },
        {
            'progress': '1/2',
            'description': 'Connect with a friend',
            'stroke_offset': 50
        }
    ]

    return render_template(
        'index.html',
        user=user_data,
        wellness=wellness_data,
        recommendations=recommendations,
        goals=goals
    )

if __name__ == '__main__':
    app.run(debug=True)
