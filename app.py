import streamlit as st
import json
import time
import subprocess
import tempfile
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import requests

# Page configuration
st.set_page_config(
    page_title="Adaptive C Programming Tutor",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #8B5CF6;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        text-align: center;
        color: #A78BFA;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #8B5CF6;
    }
    .student-message {
        background-color: #8B5CF6;
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .tutor-message {
        background-color: #1E293B;
        color: #E9D5FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .code-output {
        background-color: #0F172A;
        color: #10B981;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_user' not in st.session_state:
    st.session_state.current_user = None
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'learning_history' not in st.session_state:
    st.session_state.learning_history = []
if 'milestones' not in st.session_state:
    st.session_state.milestones = []
if 'turn_counter' not in st.session_state:
    st.session_state.turn_counter = 0
if 'demo_mode' not in st.session_state:
    st.session_state.demo_mode = True
if 'all_users_data' not in st.session_state:
    st.session_state.all_users_data = {}
if 'admin_mode' not in st.session_state:
    st.session_state.admin_mode = False
if 'code_executions' not in st.session_state:
    st.session_state.code_executions = []
if 'user_input_value' not in st.session_state:
    st.session_state.user_input_value = ""

# Admin password
ADMIN_PASSWORD = "admin123"  # Change this!

# ============================================================================
# FREE LLM API INTEGRATION (Hugging Face)
# ============================================================================
def call_free_llm(prompt, max_tokens=1000):
    """
    Call free Hugging Face API (Mistral-7B or similar)
    Get free API key at: https://huggingface.co/settings/tokens
    """
    try:
        # Try to get API key from secrets or use demo mode
        api_key = st.secrets.get("HUGGINGFACE_API_KEY", "")
        
        if not api_key:
            return "Error: No Hugging Face API key found. Please add it to secrets or use Demo Mode."
        
        # Using Mistral-7B-Instruct (free tier available)
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True
            }
        }
        
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', 'No response generated')
            return str(result)
        else:
            return f"API Error {response.status_code}: {response.text}"
            
    except Exception as e:
        return f"Error calling LLM: {str(e)}"

# Alternative: Using Groq (also free and fast)
def call_groq_llm(prompt, max_tokens=1000):
    """
    Alternative free API using Groq (very fast)
    Get free API key at: https://console.groq.com/
    """
    try:
        api_key = st.secrets.get("GROQ_API_KEY", "")
        
        if not api_key:
            return None
        
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": "mixtral-8x7b-32768",  # Fast and free
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        
        return None
            
    except Exception as e:
        return None

# ============================================================================
# CODE EXECUTION
# ============================================================================
def execute_c_code(code):
    """Execute C code and return output"""
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            c_file = os.path.join(tmpdir, "program.c")
            exe_file = os.path.join(tmpdir, "program.exe" if os.name == 'nt' else "program")
            
            # Write C code to file
            with open(c_file, 'w') as f:
                f.write(code)
            
            # Compile
            compile_result = subprocess.run(
                ['gcc', c_file, '-o', exe_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if compile_result.returncode != 0:
                return {
                    'success': False,
                    'error': f"Compilation Error:\n{compile_result.stderr}",
                    'output': None
                }
            
            # Execute
            run_result = subprocess.run(
                [exe_file],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            return {
                'success': True,
                'error': None,
                'output': run_result.stdout if run_result.stdout else "(No output)"
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': "Execution timeout (infinite loop?)",
            'output': None
        }
    except FileNotFoundError:
        return {
            'success': False,
            'error': "GCC compiler not found. Install GCC to execute C code.",
            'output': None
        }
    except Exception as e:
        return {
            'success': False,
            'error': f"Error: {str(e)}",
            'output': None
        }

# ============================================================================
# STATE ESTIMATOR CLASS
# ============================================================================
class StateEstimator:
    def estimate_knowledge(self, text):
        keywords = {
            'advanced': ['pointer arithmetic', 'malloc', 'calloc', 'realloc', 'free', 'struct', 
                        'typedef', 'file handling', 'fopen', 'fclose', 'dynamic memory', 
                        'linked list', 'recursion', 'header file', 'preprocessor', 'bitwise',
                        'union', 'enum', 'extern', 'static', 'volatile', 'const'],
            'intermediate': ['array', 'loop', 'function', 'for loop', 'while', 'if else',
                           'switch', 'case', 'break', 'continue', 'variable', 'data type', 
                           'return', 'void', 'parameter', 'argument', 'scope', 'string'],
            'basic': ['printf', 'scanf', 'int', 'char', 'float', 'double', 'main', 
                     'include', 'stdio', 'void', 'return 0']
        }
        
        lower = text.lower()
        score = 0.0
        
        for kw in keywords['advanced']:
            if kw in lower:
                score += 0.15
        for kw in keywords['intermediate']:
            if kw in lower:
                score += 0.08
        for kw in keywords['basic']:
            if kw in lower:
                score += 0.03
        
        if 'pointer' in lower and ('address' in lower or 'memory' in lower):
            score += 0.1
        if 'array' in lower and 'index' in lower:
            score += 0.05
        if '#include' in text:
            score += 0.05
            
        return min(1.0, score)
    
    def estimate_confidence(self, text):
        certain_markers = ['definitely', 'clearly', 'obviously', 'sure', 'know', 'understand', 'got it']
        uncertain_markers = ['maybe', 'perhaps', 'i think', 'not sure', 'confused', "don't understand", 
                            'help', 'stuck', 'error', 'wrong']
        
        lower = text.lower()
        score = 0.5
        
        for marker in certain_markers:
            if marker in lower:
                score += 0.1
        for marker in uncertain_markers:
            if marker in lower:
                score -= 0.15
        
        question_count = text.count('?')
        score -= question_count * 0.1
        
        return max(0.0, min(1.0, score))
    
    def classify_error(self, text):
        lower = text.lower()
        
        if any(word in lower for word in ['pointer', 'address', '&', '*', 'dereference']):
            return 'pointer_confusion'
        if any(word in lower for word in ['array', 'index', 'subscript', 'bounds']):
            return 'array_issue'
        if any(word in lower for word in ['syntax', 'semicolon', 'bracket', 'compile error', 'expected']):
            return 'syntax_error'
        if any(word in lower for word in ['memory', 'segmentation', 'malloc', 'free', 'leak']):
            return 'memory_management'
        if any(word in lower for word in ['loop', 'infinite', 'iteration', 'while']):
            return 'loop_issue'
        if any(word in lower for word in ['function', 'parameter', 'return', 'call']):
            return 'function_issue'
        if any(word in lower for word in ['string', 'char array', 'strcpy', 'strlen']):
            return 'string_issue'
        
        return 'none'
    
    def estimate_engagement(self, text):
        length = len(text.strip())
        has_question = '?' in text
        has_example = 'example' in text.lower() or 'like' in text.lower()
        has_code = any(kw in text for kw in ['int ', 'char ', 'float ', 'printf', 'scanf', '#include'])
        
        score = 0.5
        
        if length > 100:
            score += 0.2
        elif length > 50:
            score += 0.1
        elif length < 20:
            score -= 0.2
        
        if has_question:
            score += 0.15
        if has_example:
            score += 0.1
        if has_code:
            score += 0.15
        
        return max(0.0, min(1.0, score))
    
    def estimate(self, student_input):
        return {
            'k': self.estimate_knowledge(student_input),
            'c': self.estimate_confidence(student_input),
            'e': self.classify_error(student_input),
            'm': self.estimate_engagement(student_input)
        }

# ============================================================================
# POLICY CONTROLLER
# ============================================================================
class PolicyController:
    def select_action(self, state):
        k, c, e, m = state['k'], state['c'], state['e'], state['m']
        
        difficulty = 0.3 + (k * 0.6)
        speed = 0.5 + (c * 0.3) - ((1 - m) * 0.2)
        support = 0.9 - (k * 0.5) - (c * 0.3)
        
        error_adjustments = {
            'pointer_confusion': (0.2, 0.0, -0.2),
            'memory_management': (0.15, -0.2, 0.0),
            'syntax_error': (0.1, 0.0, 0.0),
            'array_issue': (0.15, -0.1, -0.1),
            'function_issue': (0.1, -0.1, 0.0),
            'string_issue': (0.1, -0.1, 0.0),
            'loop_issue': (0.1, -0.15, 0.0)
        }
        
        if e in error_adjustments:
            support_adj, speed_adj, diff_adj = error_adjustments[e]
            support += support_adj
            speed = max(0.3, speed + speed_adj)
            difficulty = max(0.2, difficulty + diff_adj)
        
        difficulty = max(0.0, min(1.0, difficulty))
        speed = max(0.0, min(1.0, speed))
        support = max(0.0, min(1.0, support))
        
        return {'difficulty': difficulty, 'speed': speed, 'support': support}

# ============================================================================
# PROMPT GENERATOR
# ============================================================================
class PromptGenerator:
    def build_prompt(self, action, student_query, history, state):
        difficulty = action['difficulty']
        speed = action['speed']
        support = action['support']
        
        difficulty_desc = "basic" if difficulty < 0.3 else "intermediate" if difficulty < 0.7 else "advanced"
        speed_desc = "detailed and step-by-step" if speed < 0.4 else "balanced" if speed < 0.7 else "concise"
        support_desc = "full explanation with code examples" if support > 0.7 else "guided hints" if support > 0.3 else "minimal hints"
        
        recent_history = []
        for msg in history[-2:]:
            role = "Student" if msg['role'] == 'student' else "Tutor"
            recent_history.append(f"{role}: {msg['content'][:150]}")
        history_text = "\n".join(recent_history) if recent_history else "Start of conversation."
        
        error_guidance_map = {
            'pointer_confusion': "Focus on pointers, addresses, * and & operators.",
            'memory_management': "Focus on malloc/free and memory safety.",
            'array_issue': "Focus on array indexing and bounds.",
            'syntax_error': "Focus on C syntax rules and common mistakes.",
            'loop_issue': "Focus on loop control and conditions.",
            'function_issue': "Focus on function parameters and return values.",
            'string_issue': "Focus on string handling and char arrays."
        }
        
        error_guidance = error_guidance_map.get(state['e'], "")
        
        prompt = f"""You are a C programming tutor. Adapt your response:
- Difficulty: {difficulty_desc} 
- Style: {speed_desc}
- Support: {support_desc}
{error_guidance}

Recent context: {history_text}

Student asks: {student_query}

Provide a helpful, adaptive response with code examples if useful. Keep it focused on C programming."""
        
        return prompt

# ============================================================================
# DEMO RESPONSES (Continued from previous version)
# ============================================================================
def generate_demo_response(state, action, user_message):
    """Generate comprehensive demo responses"""
    difficulty = action['difficulty']
    support = action['support']
    lower = user_message.lower()
    
    # Variable topic
    if 'variable' in lower:
        if support > 0.7:
            return """**Variables in C** store data in memory. You must declare type before use.

**Basic Types:**

```c
int age = 25;          // Integers (-2147483648 to 2147483647)
float price = 19.99;   // Decimals (6-7 digits precision)
char grade = 'A';      // Single character
double pi = 3.14159;   // Large decimals (15-16 digits)
```

**Example Program:**
```c
#include <stdio.h>

int main() {
    int age = 21;
    float height = 5.9;
    char grade = 'B';
    
    printf("Age: %d\\n", age);
    printf("Height: %.1f\\n", height);
    printf("Grade: %c\\n", grade);
    
    return 0;
}
```"""
        else:
            return """Variables store data. Types: `int` (whole numbers), `float` (decimals), `char` (character).
Example: `int age = 25;`"""
    
    # Pointer topic
    elif 'pointer' in lower:
        if support > 0.7:
            return """**Pointers** store memory addresses.

**Key Concepts:**
- `&` gets the address
- `*` dereferences (accesses value)

```c
int x = 10;
int *ptr = &x;  // ptr stores address of x

printf("%d", x);     // 10
printf("%d", *ptr);  // 10 (dereferencing)
printf("%p", ptr);   // 0x7fff... (address)

*ptr = 20;  // Changes x to 20
printf("%d", x);     // 20
```

Think: pointer is like a house address, dereferencing is like going inside."""
        else:
            return """Pointers store addresses. `&x` gets address, `*ptr` accesses value.
Example: `int *ptr = &x;`"""
    
    # Array topic
    elif 'array' in lower:
        if difficulty > 0.6:
            return """**Arrays** are contiguous memory blocks.

```c
int arr[5] = {10, 20, 30, 40, 50};

// Array indexing
printf("%d", arr[0]);     // 10
printf("%d", arr[2]);     // 30

// Pointer arithmetic (advanced)
printf("%d", *(arr + 2)); // 30 (same as arr[2])
```

**Key points:**
- Zero-indexed (0 to n-1)
- Fixed size
- Array name is pointer to first element"""
        else:
            return """**Arrays** store multiple values:

```c
int numbers[5] = {10, 20, 30, 40, 50};
printf("%d", numbers[0]);  // 10
printf("%d", numbers[2]);  // 30
```

Use index 0 to n-1."""
    
    # Loop topic
    elif any(word in lower for word in ['loop', 'for', 'while']):
        return """**C Loops:**

**1. for loop** (known iterations):
```c
for(int i = 0; i < 5; i++) {
    printf("%d ", i);  // 0 1 2 3 4
}
```

**2. while loop** (condition-based):
```c
int i = 0;
while(i < 5) {
    printf("%d ", i);
    i++;
}
```

**3. do-while** (runs at least once):
```c
int i = 0;
do {
    printf("%d ", i);
    i++;
} while(i < 5);
```"""
    
    # Function topic
    elif 'function' in lower:
        return """**Functions** organize reusable code blocks.

**Structure:**
```c
return_type name(parameters) {
    // code
    return value;
}
```

**Example:**
```c
// Function definition
int add(int a, int b) {
    return a + b;
}

// Function call
int main() {
    int sum = add(5, 3);  // sum = 8
    printf("Sum: %d", sum);
    return 0;
}
```"""
    
    # I/O topic
    elif any(word in lower for word in ['printf', 'scanf', 'input', 'output']):
        return """**Input/Output in C:**

```c
#include <stdio.h>

int main() {
    int age;
    float height;
    
    // Output
    printf("Enter age: ");
    
    // Input (note the &)
    scanf("%d", &age);
    
    printf("Enter height: ");
    scanf("%f", &height);
    
    // Formatted output
    printf("Age: %d, Height: %.2f\\n", age, height);
    
    return 0;
}
```

**Format specifiers:**
%d (int), %f (float), %c (char), %s (string)"""
    
    # Conditionals
    elif any(word in lower for word in ['if', 'else', 'condition', 'switch']):
        return """**Conditionals:**

```c
int age = 18;

// if-else
if (age >= 18) {
    printf("Adult");
} else if (age >= 13) {
    printf("Teenager");
} else {
    printf("Child");
}

// switch
switch(age) {
    case 18:
        printf("Just adult");
        break;
    case 21:
        printf("Twenty one");
        break;
    default:
        printf("Other age");
}
```

**Operators:** == != > < >= <="""
    
    # Struct topic
    elif 'struct' in lower:
        return """**Structures** group related data:

```c
struct Student {
    char name[50];
    int age;
    float gpa;
};

int main() {
    struct Student s1;
    s1.age = 20;
    s1.gpa = 3.8;
    strcpy(s1.name, "Alice");
    
    printf("Name: %s\\n", s1.name);
    printf("Age: %d\\n", s1.age);
    printf("GPA: %.2f\\n", s1.gpa);
    
    return 0;
}
```

Access members with dot (.)"""
    
    # Memory management
    elif any(word in lower for word in ['malloc', 'free', 'dynamic', 'memory']):
        return """**Dynamic Memory:**

```c
#include <stdlib.h>

int main() {
    // Allocate
    int *arr = (int*)malloc(5 * sizeof(int));
    
    if (arr == NULL) {
        return 1;  // Failed
    }
    
    // Use
    for(int i = 0; i < 5; i++) {
        arr[i] = i * 10;
    }
    
    // Free (important!)
    free(arr);
    return 0;
}
```

**Always free what you malloc!**"""
    
    # String topic
    elif 'string' in lower:
        return """**Strings** are char arrays ending with '\\0':

```c
#include <string.h>

char name[20] = "John";

// String functions
strlen(name);           // Length
strcpy(name, "Jane");   // Copy
strcat(name, " Doe");   // Concat
strcmp(str1, str2);     // Compare

// Input
scanf("%s", name);  // No & for arrays

// Output
printf("%s", name);
```"""
    
    # Generic response
    return f"""That's an important C concept! 

{"I'll provide detailed help with examples." if support > 0.6 else "Think about how C manages memory and types."}

Would you like me to:
- Show a code example?
- Explain specific details?
- Show common mistakes?"""

# ============================================================================
# DATA EXPORT
# ============================================================================
def export_user_data_csv(username):
    if username not in st.session_state.all_users_data:
        return None
    
    data = st.session_state.all_users_data[username]
    df = pd.DataFrame(data['learning_history'])
    df['username'] = username
    
    return df

def export_all_users_csv():
    all_data = []
    
    for username, data in st.session_state.all_users_data.items():
        for entry in data['learning_history']:
            entry_copy = entry.copy()
            entry_copy['username'] = username
            all_data.append(entry_copy)
    
    if not all_data:
        return None
    
    return pd.DataFrame(all_data)

# ============================================================================
# LEARNING TRACKER
# ============================================================================
def check_milestones(state, turn):
    milestones_config = [
        (0.3, 'first_understanding', 'k', '🎯 First signs of C understanding!'),
        (0.7, 'confident', 'c', '💪 Showing strong confidence!'),
        (0.6, 'advanced', 'k', '🚀 Engaging with advanced concepts!'),
        (0.8, 'engaged', 'm', '⚡ Highly engaged!')
    ]
    
    for threshold, m_type, key, message in milestones_config:
        if state[key] >= threshold and not any(m['type'] == m_type for m in st.session_state.milestones):
            st.session_state.milestones.append({
                'type': m_type,
                'turn': turn,
                'message': message,
                'time': datetime.now().strftime("%H:%M:%S")
            })

def get_learning_level(k):
    levels = [(0.2, 'Novice'), (0.4, 'Beginner'), (0.6, 'Intermediate'), (0.8, 'Advanced'), (1.0, 'Expert')]
    for threshold, level in levels:
        if k < threshold:
            return level
    return 'Expert'

def save_user_data():
    if st.session_state.current_user:
        username = st.session_state.current_user['username']
        st.session_state.all_users_data[username] = {
            'user_info': st.session_state.current_user,
            'messages': st.session_state.messages,
            'learning_history': st.session_state.learning_history,
            'milestones': st.session_state.milestones,
            'turn_counter': st.session_state.turn_counter,
            'code_executions': st.session_state.code_executions,
            'last_active': datetime.now().isoformat()
        }

# ============================================================================
# ADMIN DASHBOARD
# ============================================================================
def show_admin_dashboard():
    st.markdown('<div class="main-header">👨‍💼 Admin Dashboard</div>', unsafe_allow_html=True)
    
    if not st.session_state.all_users_data:
        st.info("📊 No student data available yet.")
        return
    
    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(st.session_state.all_users_data))
    with col2:
        total_interactions = sum(data['turn_counter'] for data in st.session_state.all_users_data.values())
        st.metric("Total Interactions", total_interactions)
    with col3:
        total_milestones = sum(len(data['milestones']) for data in st.session_state.all_users_data.values())
        st.metric("Total Milestones", total_milestones)
    with col4:
        total_code_exec = sum(len(data.get('code_executions', [])) for data in st.session_state.all_users_data.values())
        st.metric("Code Executions", total_code_exec)
    
    st.markdown("---")
    
    # Export
    if st.button("📥 Export All Data (CSV)", use_container_width=False):
        df = export_all_users_csv()
        if df is not None:
            csv = df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"all_students_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
    
    # Student comparison
    st.markdown("### 📊 Student Comparison")
    
    comparison_data = []
    for username, data in st.session_state.all_users_data.items():
        if data['learning_history']:
            latest = data['learning_history'][-1]
            first = data['learning_history'][0]
            comparison_data.append({
                'Student': username,
                'Level': get_learning_level(latest['k']),
                'Knowledge': latest['k'],
                'Confidence': latest['c'],
                'Engagement': latest['m'],
                'Growth': latest['k'] - first['k'],
                'Turns': data['turn_counter'],
                'Milestones': len(data['milestones'])
            })
    
    if comparison_data:
        df_comparison = pd.DataFrame(comparison_data)
        st.dataframe(df_comparison, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(df_comparison, x='Student', y='Knowledge', 
                        title='Knowledge by Student',
                        color='Knowledge')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(df_comparison, x='Student', y='Growth', 
                        title='Growth by Student',
                        color='Growth')
            st.plotly_chart(fig, use_container_width=True)
        
        # Trajectories
        st.markdown("### 📈 Learning Trajectories")
        fig = go.Figure()
        for username, data
