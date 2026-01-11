"""
COMPLETE C PROGRAMMING TUTOR - STREAMLIT APP
Save this entire file as: app.py
"""

import streamlit as st
import time
import subprocess
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests

# Page config
st.set_page_config(page_title="C Programming Tutor", page_icon="💻", layout="wide")

# CSS
st.markdown("""
<style>
.student-message {background-color: #8B5CF6; color: white; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
.tutor-message {background-color: #1E293B; color: #E9D5FF; padding: 1rem; border-radius: 0.5rem; margin: 0.5rem 0;}
.code-output {background-color: #0F172A; color: #10B981; padding: 1rem; border-radius: 0.5rem; font-family: monospace;}
</style>
""", unsafe_allow_html=True)

# Session state
for key, default in [('current_user', None), ('messages', []), ('learning_history', []), ('milestones', []), 
                     ('turn_counter', 0), ('demo_mode', True), ('all_users_data', {}), ('admin_mode', False), 
                     ('code_executions', []), ('admin_authenticated', False)]:
    if key not in st.session_state:
        st.session_state[key] = default

ADMIN_PASSWORD = "admin123"

# LLM API calls
def call_groq_llm(prompt, max_tokens=800):
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "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")
        if not api_key:
            return None
        response = requests.post("https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": "mixtral-8x7b-32768", "messages": [{"role": "user", "content": prompt}], 
                  "max_tokens": max_tokens, "temperature": 0.7}, timeout=30)
        if response.status_code == 200:
            return response.json()['choices'][0]['message']['content']
    except: pass
    return None

def call_huggingface_llm(prompt, max_tokens=800):
    try:
        api_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
        if not api_key:
            return None
        response = requests.post("https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2",
            headers={"Authorization": f"Bearer {api_key}"}, 
            json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "temperature": 0.7}}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            return result[0]['generated_text'] if isinstance(result, list) else str(result)
    except: pass
    return None

# Code execution
def execute_c_code(code):
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "program.c")
            exe_file = os.path.join(tmpdir, "program.exe" if os.name == 'nt' else "program")
            with open(c_file, 'w') as f:
                f.write(code)
            compile_result = subprocess.run(['gcc', c_file, '-o', exe_file], capture_output=True, text=True, timeout=5)
            if compile_result.returncode != 0:
                return {'success': False, 'error': f"Compilation Error:\n{compile_result.stderr}"}
            run_result = subprocess.run([exe_file], capture_output=True, text=True, timeout=5)
            return {'success': True, 'output': run_result.stdout or "(No output)"}
    except subprocess.TimeoutExpired:
        return {'success': False, 'error': "Timeout (infinite loop?)"}
    except FileNotFoundError:
        return {'success': False, 'error': "GCC not found. Install GCC compiler."}
    except Exception as e:
        return {'success': False, 'error': str(e)}

# State estimator
class StateEstimator:
    def estimate(self, text):
        lower = text.lower()
        k = sum(0.15 if kw in lower else 0 for kw in ['pointer', 'malloc', 'struct', 'typedef', 'recursion'])
        k += sum(0.08 if kw in lower else 0 for kw in ['array', 'loop', 'function', 'variable'])
        k += sum(0.03 if kw in lower else 0 for kw in ['printf', 'scanf', 'int', 'char'])
        k = min(1.0, k)
        
        c = 0.5
        c += sum(0.1 if m in lower else 0 for m in ['sure', 'know', 'understand'])
        c -= sum(0.15 if m in lower else 0 for m in ['confused', 'not sure', 'help'])
        c -= text.count('?') * 0.1
        c = max(0, min(1.0, c))
        
        e = 'pointer_confusion' if 'pointer' in lower else 'array_issue' if 'array' in lower else \
            'syntax_error' if 'syntax' in lower else 'memory_management' if 'malloc' in lower else 'none'
        
        m = 0.5
        length = len(text.strip())
        m += 0.2 if length > 100 else 0.1 if length > 50 else -0.2 if length < 20 else 0
        m += 0.15 if '?' in text else 0
        m = max(0, min(1.0, m))
        
        return {'k': k, 'c': c, 'e': e, 'm': m}

# Policy controller
class PolicyController:
    def select_action(self, state):
        k, c, m = state['k'], state['c'], state['m']
        diff = max(0, min(1, 0.3 + k * 0.6))
        speed = max(0, min(1, 0.5 + c * 0.3 - (1 - m) * 0.2))
        support = max(0, min(1, 0.9 - k * 0.5 - c * 0.3))
        return {'difficulty': diff, 'speed': speed, 'support': support}

# Prompt generator
class PromptGenerator:
    def build_prompt(self, action, query, history, state):
        d = "basic" if action['difficulty'] < 0.3 else "intermediate" if action['difficulty'] < 0.7 else "advanced"
        s = "detailed" if action['speed'] < 0.4 else "balanced" if action['speed'] < 0.7 else "concise"
        sup = "full examples" if action['support'] > 0.7 else "guided hints" if action['support'] > 0.3 else "minimal"
        hist = "\n".join([f"{'Student' if m['role']=='student' else 'Tutor'}: {m['content'][:100]}" for m in history[-2:]])
        return f"""C programming tutor. Adapt response: Difficulty={d}, Style={s}, Support={sup}
Context: {hist}
Student: {query}
Provide helpful C programming response with code examples."""

# Demo responses
def generate_demo_response(state, action, msg):
    lower = msg.lower()
    support = action['support']
    
    if 'variable' in lower:
        return """**Variables in C:**
```c
int age = 25;        // Integer
float price = 19.99; // Decimal
char grade = 'A';    // Character
```
Types: int, float, char, double""" if support > 0.7 else "Variables store data. Use: int, float, char"
    
    if 'pointer' in lower:
        return """**Pointers:**
```c
int x = 10;
int *ptr = &x;  // ptr stores address of x
printf("%d", *ptr);  // 10
```
& gets address, * accesses value""" if support > 0.7 else "Pointers store addresses. & gets address, * accesses value"
    
    if 'array' in lower:
        return """**Arrays:**
```c
int arr[5] = {10, 20, 30, 40, 50};
printf("%d", arr[0]);  // 10
```
Zero-indexed collection""" if support > 0.7 else "Arrays store multiple values: int arr[5] = {1,2,3,4,5};"
    
    if any(w in lower for w in ['loop', 'for', 'while']):
        return """**Loops:**
```c
for(int i = 0; i < 5; i++) {
    printf("%d ", i);
}

while(i < 5) {
    printf("%d ", i);
    i++;
}
```"""
    
    if 'function' in lower:
        return """**Functions:**
```c
int add(int a, int b) {
    return a + b;
}

int main() {
    int sum = add(5, 3);  // 8
    return 0;
}
```"""
    
    return "That's an important C concept! Would you like a code example or detailed explanation?"

# Learning tracker
def check_milestones(state, turn):
    configs = [(0.3, 'first', 'k', '🎯 First understanding!'), (0.7, 'conf', 'c', '💪 Confident!'),
               (0.6, 'adv', 'k', '🚀 Advanced!'), (0.8, 'eng', 'm', '⚡ Engaged!')]
    for thresh, typ, key, msg in configs:
        if state[key] >= thresh and not any(m['type'] == typ for m in st.session_state.milestones):
            st.session_state.milestones.append({'type': typ, 'turn': turn, 'message': msg, 'time': datetime.now().strftime("%H:%M")})

def get_level(k):
    return 'Novice' if k < 0.2 else 'Beginner' if k < 0.4 else 'Intermediate' if k < 0.6 else 'Advanced' if k < 0.8 else 'Expert'

def save_user_data():
    if st.session_state.current_user:
        u = st.session_state.current_user['username']
        st.session_state.all_users_data[u] = {
            'user_info': st.session_state.current_user, 'messages': st.session_state.messages,
            'learning_history': st.session_state.learning_history, 'milestones': st.session_state.milestones,
            'turn_counter': st.session_state.turn_counter, 'code_executions': st.session_state.code_executions
        }

# Export
def export_csv(username=None):
    if username:
        if username not in st.session_state.all_users_data: return None
        df = pd.DataFrame(st.session_state.all_users_data[username]['learning_history'])
        df['username'] = username
        return df
    all_data = []
    for u, d in st.session_state.all_users_data.items():
        for e in d['learning_history']:
            e_copy = e.copy()
            e_copy['username'] = u
            all_data.append(e_copy)
    return pd.DataFrame(all_data) if all_data else None

# Admin dashboard
def show_admin():
    st.markdown("# 👨‍💼 Admin Dashboard")
    if not st.session_state.all_users_data:
        st.info("No data yet")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Students", len(st.session_state.all_users_data))
    col2.metric("Interactions", sum(d['turn_counter'] for d in st.session_state.all_users_data.values()))
    col3.metric("Milestones", sum(len(d['milestones']) for d in st.session_state.all_users_data.values()))
    col4.metric("Code Runs", sum(len(d.get('code_executions', [])) for d in st.session_state.all_users_data.values()))
    
    if st.button("📥 Export All CSV"):
        df = export_csv()
        if df is not None:
            st.download_button("Download", df.to_csv(index=False), f"all_students.csv", "text/csv")
    
    st.markdown("### Students")
    data = []
    for u, d in st.session_state.all_users_data.items():
        if d['learning_history']:
            l = d['learning_history'][-1]
            data.append({'Student': u, 'Level': get_level(l['k']), 'Knowledge': l['k'], 
                        'Turns': d['turn_counter'], 'Milestones': len(d['milestones'])})
    if data:
        st.dataframe(pd.DataFrame(data), use_container_width=True)
        fig = go.Figure()
        for u, d in st.session_state.all_users_data.items():
            if d['learning_history']:
                turns = [h['turn'] for h in d['learning_history']]
                k = [h['k']*100 for h in d['learning_history']]
                fig.add_trace(go.Scatter(x=turns, y=k, mode='lines+markers', name=u))
        fig.update_layout(title="Progress", xaxis_title="Turn", yaxis_title="Knowledge (%)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

# Main app
def main():
    with st.sidebar:
        st.markdown("### 👨‍💼 Admin")
        pw = st.text_input("Password", type="password")
        if st.button("Login"):
            if pw == ADMIN_PASSWORD:
                st.session_state.admin_authenticated = True
                st.session_state.admin_mode = True
                st.rerun()
        if st.session_state.admin_authenticated and st.button("Student Mode"):
            st.session_state.admin_mode = False
            st.rerun()
    
    if st.session_state.admin_mode and st.session_state.admin_authenticated:
        show_admin()
        return
    
    if not st.session_state.current_user:
        st.markdown("# 💻 C Programming Tutor")
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.info("Enter username to start")
            username = st.text_input("Username")
            if st.button("🚀 Start", use_container_width=True):
                if username.strip():
                    st.session_state.current_user = {'username': username.strip(), 'created_at': datetime.now().isoformat()}
                    if username.strip() in st.session_state.all_users_data:
                        saved = st.session_state.all_users_data[username.strip()]
                        st.session_state.messages = saved.get('messages', [])
                        st.session_state.learning_history = saved.get('learning_history', [])
                        st.session_state.milestones = saved.get('milestones', [])
                        st.session_state.turn_counter = saved.get('turn_counter', 0)
                    st.rerun()
        return
    
    user = st.session_state.current_user
    
    with st.sidebar:
        st.markdown(f"### 👤 {user['username']}")
        st.markdown("---")
        st.session_state.demo_mode = st.toggle("🎮 Demo", st.session_state.demo_mode)
        st.markdown("---")
        
        if st.session_state.learning_history:
            st.markdown("### 📊 State")
            l = st.session_state.learning_history[-1]
            st.metric("Knowledge", f"{l['k']:.2f}")
            st.progress(l['k'])
            st.metric("Confidence", f"{l['c']:.2f}")
            st.progress(l['c'])
            st.metric("Engagement", f"{l['m']:.2f}")
            st.progress(l['m'])
            st.markdown(f"**Level:** {get_level(l['k'])}")
        
        st.markdown("---")
        if st.session_state.milestones:
            st.markdown("### 🏆 Milestones")
            for m in st.session_state.milestones[-3:]:
                st.success(f"{m['message']}\nTurn {m['turn']}")
        
        st.markdown("---")
        if st.button("🔄 New", use_container_width=True):
            save_user_data()
            st.session_state.messages = []
            st.session_state.learning_history = []
            st.session_state.milestones = []
            st.session_state.turn_counter = 0
            st.rerun()
        if st.button("💾 Save", use_container_width=True):
            save_user_data()
            st.success("Saved!")
        if st.button("🚪 Logout", use_container_width=True):
            save_user_data()
            st.session_state.current_user = None
            st.rerun()
    
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "💻 Code", "📊 Progress"])
    
    with tab1:
        st.markdown(f"### Welcome, {user['username']}!")
        if not st.session_state.messages:
            st.info("Ask: 'Explain variables' or 'What are pointers?'")
        
        for msg in st.session_state.messages:
            if msg['role'] == 'student':
                st.markdown(f'<div class="student-message"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tutor-message"><b>Tutor:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
        
        with st.form(key='chat', clear_on_submit=True):
            inp = st.text_area("Ask:", height=100)
            if st.form_submit_button("Send 📤"):
                if inp.strip():
                    estimator = StateEstimator()
                    policy = PolicyController()
                    prompter = PromptGenerator()
                    
                    st.session_state.messages.append({'role': 'student', 'content': inp, 'timestamp': datetime.now().isoformat()})
                    st.session_state.turn_counter += 1
                    state = estimator.estimate(inp)
                    state['turn'] = st.session_state.turn_counter
                    st.session_state.learning_history.append(state)
                    check_milestones(state, st.session_state.turn_counter)
                    action = policy.select_action(state)
                    
                    with st.spinner("Thinking..."):
                        if st.session_state.demo_mode:
                            time.sleep(0.5)
                            response = generate_demo_response(state, action, inp)
                        else:
                            prompt = prompter.build_prompt(action, inp, st.session_state.messages, state)
                            response = call_groq_llm(prompt) or call_huggingface_llm(prompt) or "API error. Enable Demo Mode."
                    
                    st.session_state.messages.append({'role': 'tutor', 'content': response, 'timestamp': datetime.now().isoformat()})
                    save_user_data()
                    st.rerun()
    
    with tab2:
        st.markdown("### 💻 Code Execution")
        code = st.text_area("C Code:", height=300, value='#include <stdio.h>\n\nint main() {\n    printf("Hello!\\n");\n    return 0;\n}')
        if st.button("▶️ Run"):
            if code.strip():
                with st.spinner("Compiling..."):
                    result = execute_c_code(code)
                    st.session_state.code_executions.append({'timestamp': datetime.now().isoformat(), 'code': code, 
                                                             'success': result['success'], 
                                                             'output': result.get('output') or result.get('error')})
                    save_user_data()
                    if result['success']:
                        st.success("✅ Success!")
                        st.markdown(f'<div class="code-output">{result["output"]}</div>', unsafe_allow_html=True)
                    else:
                        st.error("❌ Failed")
                        st.code(result['error'])
    
    with tab3:
        st.markdown("### 📊 Progress")
        if not st.session_state.learning_history:
            st.info("Start chatting to see progress")
            return
        
        l = st.session_state.learning_history[-1]
        f = st.session_state.learning_history[0]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Level", get_level(l['k']))
        col2.metric("Turns", st.session_state.turn_counter)
        col3.metric("Growth", f"{(l['k']-f['k'])*100:+.0f}%")
        col4.metric("Milestones", len(st.session_state.milestones))
        
        turns = [h['turn'] for h in st.session_state.learning_history]
        k = [h['k']*100 for h in st.session_state.learning_history]
        c = [h['c']*100 for h in st.session_state.learning_history]
        m = [h['m']*100 for h in st.session_state.learning_history]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=turns, y=k, mode='lines+markers', name='Knowledge', line=dict(color='#3B82F6', width=3)))
        fig.add_trace(go.Scatter(x=turns, y=c, mode='lines+markers', name='Confidence', line=dict(color='#06B6D4', width=3)))
        fig.add_trace(go.Scatter(x=turns, y=m, mode='lines+markers', name='Engagement', line=dict(color='#EC4899', width=3)))
        fig.update_layout(xaxis_title="Turn", yaxis_title="Score (%)", template="plotly_dark", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        if st.button("📥 Export CSV"):
            df = export_csv(user['username'])
            if df is not None:
                st.download_button("Download", df.to_csv(index=False), f"{user['username']}_data.csv", "text/csv")

if __name__ == "__main__":
    main()

