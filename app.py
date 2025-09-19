import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the actual Granite 3.2-2B model
model_name = "ibm-granite/granite-3.2-2b-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def generate_response(prompt, max_length=512, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)

    if torch.cuda.is_available():
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove the prompt from the response
    response = response.replace(prompt, "").strip()
    return response

def concept_explanation(concept, level="Intermediate", examples=True):
    prompt = f"Explain the concept of {concept} at a {level} level"
    if examples:
        prompt += " with practical examples and applications"
    prompt += ":"
    
    return generate_response(prompt, max_length=800)

def quiz_generator(topic, difficulty="medium", question_count=5):
    prompt = f"Generate {question_count} {difficulty} difficulty quiz questions about {topic} with different question types (multiple choice, true/false, short answer). At the end, provide all the answers in a separate ANSWERS section:"
    return generate_response(prompt, max_length=1000)

def evaluate_answers(answers, topic):
    prompt = f"Evaluate these answers to a quiz about {topic} and provide a score with feedback:\n{answers}\n\nEvaluation:"
    return generate_response(prompt, max_length=600)

def diagnostic_test(grade_level, subjects):
    prompt = f"Create a diagnostic test for a {grade_level} student focusing on {', '.join(subjects)}. Include questions that assess fundamental knowledge and identify learning gaps:"
    return generate_response(prompt, max_length=1200)

def generate_study_plan(weak_areas, timeframe="1 week"):
    prompt = f"Create a {timeframe} study plan to improve knowledge in these areas: {', '.join(weak_areas)}. Include daily topics, resources, and practice activities:"
    return generate_response(prompt, max_length=800)

def sync_google_classroom():
    # Simulate API call to Google Classroom
    time.sleep(2)
    return "✅ Successfully synchronized with Google Classroom! Retrieved 4 courses and 28 assignments."

def analyze_performance(quiz_results):
    prompt = f"Analyze these quiz results and provide insights on student performance, including strengths, weaknesses, and recommendations:\n{quiz_results}\n\nAnalysis:"
    return generate_response(prompt, max_length=800)

# Sample data for demonstration
sample_courses = ["Mathematics 101", "Physics Fundamentals", "Computer Science Principles", "History of Science"]
sample_assignments = [
    {"course": "Mathematics 101", "name": "Algebra Homework", "due": "2023-10-15", "status": "Submitted"},
    {"course": "Physics Fundamentals", "name": "Newton's Laws Lab", "due": "2023-10-18", "status": "In Progress"},
    {"course": "Computer Science Principles", "name": "Python Programming Exercise", "due": "2023-10-20", "status": "Not Started"}
]

sample_performance = {
    "Mathematics 101": {"last_score": 88, "average": 85, "quizzes_taken": 5},
    "Physics Fundamentals": {"last_score": 92, "average": 87, "quizzes_taken": 3},
    "Computer Science Principles": {"last_score": 95, "average": 90, "quizzes_taken": 4}
}

# Create Gradio interface with purple theme
purple_theme = gr.themes.Default(
    primary_hue="purple",
    secondary_hue="purple",
).set(
    body_background_fill='#f5f0ff',
    body_background_fill_dark='#1e1e2e',
    button_primary_background_fill='#8a2be2',
    button_primary_background_fill_dark='#8a2be2',
    button_primary_text_color='white',
    button_primary_text_color_dark='white',
)

with gr.Blocks(theme=purple_theme, title="EduTutor AI") as app:
    gr.Markdown(
        """
        # 🎓 EduTutor AI: Personalized Learning Assistant
        ### *Powered by IBM Granite 3.2-2B Model*
        """
    )
    
    with gr.Tabs():
        with gr.TabItem("🏠 Dashboard"):
            gr.Markdown("## Welcome to Your Learning Dashboard")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### Student Profile")
                    gr.Markdown("""
                    **Name:** Alex Johnson  
                    **Grade:** 10th  
                    **Last Login:** Today  
                    **Current Streak:** 5 days
                    """)
                    
                    gr.Markdown("### Recent Activity")
                    gr.Markdown("""
                    - Completed Quiz: Machine Learning Basics (85%)  
                    - Studied: Python Functions  
                    - Earned: Data Science Badge
                    """)
                    
                with gr.Column(scale=2):
                    gr.Markdown("### Performance Overview")
                    
                    # Performance chart
                    with gr.Row():
                        subjects = ["Math", "Science", "History", "English"]
                        scores = [85, 92, 78, 88]
                        with gr.Column():
                            for subject, score in zip(subjects, scores):
                                gr.Markdown(f"**{subject}**: {score}%")
                                gr.Slider(minimum=0, maximum=100, value=score, interactive=False)
        
        with gr.TabItem("📚 Learn"):
            gr.Markdown("## Learn New Concepts")
            
            with gr.Row():
                with gr.Column():
                    concept_input = gr.Textbox(label="Enter a concept to learn", placeholder="e.g., machine learning, photosynthesis, quantum mechanics")
                    with gr.Row():
                        explain_btn = gr.Button("Explain Concept", variant="primary")
                        clear_btn = gr.Button("Clear")
                
                with gr.Column():
                    level = gr.Radio(["Basic", "Intermediate", "Advanced"], label="Explanation Level", value="Intermediate")
                    examples = gr.Checkbox(label="Include practical examples", value=True)
            
            explanation_output = gr.Textbox(label="Explanation", lines=10)
            
            gr.Examples(
                examples=["machine learning", "physics", "python programming", "neural networks"],
                inputs=concept_input
            )
            
            explain_btn.click(concept_explanation, inputs=[concept_input, level, examples], outputs=explanation_output)
            clear_btn.click(lambda: ("", ""), outputs=[concept_input, explanation_output])
        
        with gr.TabItem("📝 Quiz"):
            gr.Markdown("## Generate Practice Quizzes")
            
            with gr.Row():
                with gr.Column():
                    quiz_input = gr.Textbox(label="Enter a topic for quiz", placeholder="e.g., physics, world history, python programming")
                    with gr.Row():
                        quiz_btn = gr.Button("Generate Quiz", variant="primary")
                        clear_quiz_btn = gr.Button("Clear")
                
                with gr.Column():
                    quiz_type = gr.Radio(["Multiple Choice", "Mixed Types", "True/False", "Short Answer"], label="Quiz Type", value="Mixed Types")
                    question_count = gr.Slider(3, 10, value=5, step=1, label="Number of Questions")
                    difficulty = gr.Radio(["Easy", "Medium", "Hard"], label="Difficulty Level", value="Medium")
            
            quiz_output = gr.Textbox(label="Generated Quiz", lines=15)
            
            gr.Examples(
                examples=["machine learning", "physics", "python programming", "calculus"],
                inputs=quiz_input
            )
            
            quiz_btn.click(quiz_generator, inputs=[quiz_input, difficulty, question_count], outputs=quiz_output)
            clear_quiz_btn.click(lambda: ("", ""), outputs=[quiz_input, quiz_output])
        
        with gr.TabItem("✅ Evaluate"):
            gr.Markdown("## Evaluate Your Answers")
            
            with gr.Row():
                with gr.Column():
                    answer_input = gr.Textbox(label="Paste your quiz answers here", placeholder="Enter your answers for evaluation...", lines=6)
                    with gr.Row():
                        eval_btn = gr.Button("Evaluate Answers", variant="primary")
                        clear_eval_btn = gr.Button("Clear")
                
                with gr.Column():
                    quiz_topic = gr.Textbox(label="Quiz Topic", placeholder="What topic was this quiz about?")
            
            eval_output = gr.Textbox(label="Evaluation Results", lines=8)
            
            eval_btn.click(evaluate_answers, inputs=[answer_input, quiz_topic], outputs=eval_output)
            clear_eval_btn.click(lambda: ("", "", ""), outputs=[answer_input, quiz_topic, eval_output])
        
        with gr.TabItem("🔍 Diagnostic"):
            gr.Markdown("## Generate Diagnostic Test")
            
            with gr.Row():
                with gr.Column():
                    grade_level = gr.Dropdown(["Elementary", "Middle School", "High School", "College"], label="Grade Level", value="High School")
                    subjects = gr.CheckboxGroup(["Math", "Science", "History", "English", "Computer Science"], label="Subjects", value=["Math", "Science"])
                    with gr.Row():
                        diag_btn = gr.Button("Generate Diagnostic Test", variant="primary")
                        clear_diag_btn = gr.Button("Clear")
                
                with gr.Column():
                    test_length = gr.Radio(["Short (15 questions)", "Standard (30 questions)", "Comprehensive (50 questions)"], label="Test Length", value="Standard (30 questions)")
            
            diag_output = gr.Textbox(label="Diagnostic Test", lines=10)
            
            diag_btn.click(diagnostic_test, inputs=[grade_level, subjects], outputs=diag_output)
            clear_diag_btn.click(lambda: (None, None, ""), outputs=[grade_level, subjects, diag_output])
        
        with gr.TabItem("📊 Analytics"):
            gr.Markdown("## Your Learning Progress")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Performance Trends")
                    
                    # Simulated performance data
                    weeks = ['Week 1', 'Week 2', 'Week 3', 'Week 4', 'Week 5']
                    scores = [65, 72, 80, 85, 88]
                    
                    # Create a simple bar chart using HTML
                    chart_html = """
                    <div style='background: white; padding: 20px; border-radius: 10px;'>
                        <h4>Weekly Performance</h4>
                        <div style='display: flex; height: 200px; align-items: flex-end; gap: 15px;'>
                    """
                    
                    for week, score in zip(weeks, scores):
                        height = score * 2  # Scale for visualization
                        chart_html += f"""
                            <div style='display: flex; flex-direction: column; align-items: center;'>
                                <div style='background: #8a2be2; width: 30px; height: {height}px; border-radius: 5px 5px 0 0;'></div>
                                <div style='margin-top: 10px;'>{week}</div>
                                <div>{score}%</div>
                            </div>
                        """
                    
                    chart_html += "</div></div>"
                    gr.HTML(chart_html)
                
                with gr.Column():
                    gr.Markdown("### Subject Breakdown")
                    
                    subjects = ["Mathematics", "Science", "History", "English"]
                    mastery = [85, 92, 78, 88]
                    
                    for subject, score in zip(subjects, mastery):
                        gr.Markdown(f"**{subject}**")
                        gr.Slider(minimum=0, maximum=100, value=score, interactive=False)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Strengths")
                    gr.Markdown("- Scientific reasoning  \n- Data analysis  \n- Problem decomposition")
                
                with gr.Column():
                    gr.Markdown("### Areas for Improvement")
                    gr.Markdown("- Historical timelines  \n- Literary analysis  \n- Writing concisely")
            
            weak_areas = gr.CheckboxGroup(["Historical timelines", "Literary analysis", "Writing concisely"], label="Select areas to improve:")
            timeframe = gr.Dropdown(["1 week", "2 weeks", "1 month"], label="Study Plan Duration", value="1 week")
            plan_btn = gr.Button("Generate Study Plan", variant="primary")
            study_plan = gr.Textbox(label="Personalized Study Plan", lines=6)
            
            plan_btn.click(generate_study_plan, inputs=[weak_areas, timeframe], outputs=study_plan)
        
        with gr.TabItem("🏫 Google Classroom"):
            gr.Markdown("## Google Classroom Integration")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Sync with Google Classroom")
                    sync_btn = gr.Button("Sync Now", variant="primary")
                    sync_status = gr.Textbox(label="Sync Status", interactive=False)
                
                with gr.Column():
                    gr.Markdown("### Connected Account")
                    gr.Markdown("""
                    **Email:** student.example@school.edu  
                    **Last Sync:** Today at 10:30 AM  
                    **Status:** Connected
                    """)
            
            gr.Markdown("### Your Courses")
            for course in sample_courses:
                with gr.Group():
                    gr.Markdown(f"**{course}**")
                    # Show assignments for this course
                    course_assignments = [a for a in sample_assignments if a["course"] == course]
                    for assignment in course_assignments:
                        gr.Markdown(f"- {assignment['name']} (Due: {assignment['due']}) - {assignment['status']}")
            
            sync_btn.click(sync_google_classroom, outputs=sync_status)
        
        with gr.TabItem("👨‍ Educator View"):
            gr.Markdown("## Educator Dashboard")
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Class Performance Overview")
                    
                    # Simulated class data
                    students = ["Alex Johnson", "Maria Garcia", "James Wilson", "Sarah Chen", "Michael Brown"]
                    avg_scores = [88, 92, 76, 85, 79]
                    
                    for student, score in zip(students, avg_scores):
                        gr.Markdown(f"**{student}**")
                        gr.Slider(minimum=0, maximum=100, value=score, interactive=False)
                
                with gr.Column():
                    gr.Markdown("### Quiz Analytics")
                    
                    quiz_results = gr.Textbox(label="Paste quiz results to analyze", lines=3)
                    analyze_btn = gr.Button("Analyze Performance", variant="primary")
                    analysis_output = gr.Textbox(label="Performance Analysis", lines=6)
                    
                    analyze_btn.click(analyze_performance, inputs=quiz_results, outputs=analysis_output)
            
            gr.Markdown("### Student Progress Reports")
            gr.Dataframe(
                value=pd.DataFrame({
                    "Student": students,
                    "Last Quiz Score": [88, 92, 76, 85, 79],
                    "Average Score": [85, 90, 72, 82, 75],
                    "Quizzes Taken": [5, 7, 3, 6, 4],
                    "Last Activity": ["Today", "Yesterday", "2 days ago", "Today", "3 days ago"]
                }),
                interactive=False
            )

    gr.Markdown("---")
    gr.Markdown("EduTutor AI • Powered by IBM Granite 3.2-2B • © 2023")

# Launch the application
if __name__ == "__main__":
    app.launch(share=True)
