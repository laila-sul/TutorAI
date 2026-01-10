import streamlit as st
import json
import time
from datetime import datetime
import plotly.graph_objects as go
import anthropic

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

# ============================================================================
# STATE ESTIMATOR CLASS
# ============================================================================
class StateEstimator:
    def estimate_knowledge(self, text):
        """Estimate knowledge based on C programming keywords"""
        keywords = {
            'advanced': ['pointer', 'malloc', 'free', 'struct', 'typedef', 'file handling', 
                        'dynamic memory', 'linked list', 'recursion', 'header file'],
            'intermediate': ['array', 'loop', 'function', 'for loop', 'while', 'if else',
                           'switch', 'variable', 'data type', 'return'],
            'basic': ['printf', 'scanf', 'int', 'char', 'float', 'main', 'include', 'stdio']
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
        
        # Bonus for correct understanding
        if 'pointer' in lower and ('address' in lower or 'memory' in lower):
            score += 0.1
        if 'array' in lower and 'index' in lower:
            score += 0.05
            
        return min(1.0, score)
    
    def estimate_confidence(self, text):
        """Estimate confidence from linguistic markers"""
        certain_markers = ['definitely', 'clearly', 'obviously', 'sure', 'know', 'understand']
        uncertain_markers = ['maybe', 'perhaps', 'i think', 'not sure', 'confused', "don't understand"]
        
        lower = text.lower()
        score = 0.5  # neutral baseline
        
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
        """Classify the type of error or confusion"""
        lower = text.lower()
        
        if any(word in lower for word in ['pointer', 'address', '&', '*']):
            return 'pointer_confusion'
        if any(word in lower for word in ['array', 'index', 'subscript']):
            return 'array_issue'
        if any(word in lower for word in ['syntax', 'semicolon', 'bracket', 'compile error']):
            return 'syntax_error'
        if any(word in lower for word in ['memory', 'segmentation', 'malloc', 'free']):
            return 'memory_management'
        if any(word in lower for word in ['loop', 'infinite', 'iteration']):
            return 'loop_issue'
        
        return 'none'
    
    def estimate_engagement(self, text):
        """Estimate engagement from response characteristics"""
        length = len(text.strip())
        has_question = '?' in text
        has_example = 'example' in text.lower() or 'like' in text.lower()
        
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
        
        return max(0.0, min(1.0, score))
    
    def estimate(self, student_input):
        """Main estimation function"""
        return {
            'k': self.estimate_knowledge(student_input),
            'c': self.estimate_confidence(student_input),
            'e': self.classify_error(student_input),
            'm': self.estimate_engagement(student_input)
        }

# ============================================================================
# POLICY CONTROLLER CLASS
# ============================================================================
class PolicyController:
    def select_action(self, state):
        """Select pedagogical actions based on learner state"""
        k, c, e, m = state['k'], state['c'], state['e'], state['m']
        
        # Continuous policy mapping
        difficulty = 0.3 + (k * 0.6)
        speed = 0.5 + (c * 0.3) - ((1 - m) * 0.2)
        support = 0.9 - (k * 0.5) - (c * 0.3)
        
        # Error-specific adjustments
        if e == 'pointer_confusion':
            support += 0.2
            difficulty = max(0.2, difficulty - 0.2)
        elif e == 'memory_management':
            support += 0.15
            speed = max(0.3, speed - 0.2)
        elif e == 'syntax_error':
            support += 0.1
        
        # Clip to [0, 1]
        difficulty = max(0.0, min(1.0, difficulty))
        speed = max(0.0, min(1.0, speed))
        support = max(0.0, min(1.0, support))
        
        return {
            'difficulty': difficulty,
            'speed': speed,
            'support': support
        }

# ============================================================================
# PROMPT GENERATOR CLASS
# ============================================================================
class PromptGenerator:
    def build_prompt(self, action, student_query, history, state):
        """Build adaptive prompt for LLM"""
        difficulty = action['difficulty']
        speed = action['speed']
        support = action['support']
        
        difficulty_desc = "basic" if difficulty < 0.3 else "intermediate" if difficulty < 0.7 else "advanced"
        speed_desc = "detailed and step-by-step" if speed < 0.4 else "balanced" if speed < 0.7 else "concise"
        support_desc = "full explanation with code examples" if support > 0.7 else "guided hints" if support > 0.3 else "minimal hints"
        
        # Format recent history
        recent_history = []
        for msg in history[-3:]:
            role = "Student" if msg['role'] == 'student' else "Tutor"
            recent_history.append(f"{role}: {msg['content']}")
        history_text = "\n".join(recent_history) if recent_history else "This is the start of the conversation."
        
        # Error-specific guidance
        error_guidance = ""
        if state['e'] == 'pointer_confusion':
            error_guidance = "\nFocus on: Explaining pointers, addresses, and the difference between * and & operators."
        elif state['e'] == 'memory_management':
            error_guidance = "\nFocus on: Dynamic memory allocation with malloc/free and memory safety."
        elif state['e'] == 'array_issue':
            error_guidance = "\nFocus on: Array indexing, bounds, and the relationship between arrays and pointers."
        elif state['e'] == 'syntax_error':
            error_guidance = "\nFocus on: C syntax rules, common mistakes, and how to read compiler errors."
        elif state['e'] == 'loop_issue':
            error_guidance = "\nFocus on: Loop control, conditions, and common loop patterns in C."
        
        prompt = f"""You are an adaptive C programming tutor. Adjust your teaching style based on these parameters:
- Difficulty Level: {difficulty_desc} ({difficulty:.2f})
- Response Style: {speed_desc} ({speed:.2f})
- Support Level: {support_desc} ({support:.2f}){error_guidance}

Recent Conversation:
{history_text}

Student's Current Question: {student_query}

Provide an adaptive tutoring response that matches the specified difficulty, pacing, and support level. Use code examples when helpful. Keep responses focused on C programming concepts."""
        
        return prompt

# ============================================================================
# DEMO RESPONSE GENERATOR
# ============================================================================
def generate_demo_response(state, action, user_message):
    """Generate simulated responses for demo mode"""
    difficulty = action['difficulty']
    support = action['support']
    lower = user_message.lower()
    
    # Topic-specific responses
    if 'pointer' in lower:
        if support > 0.7:
            return """A pointer is a variable that stores the memory address of another variable. Think of it like a house address - it tells you where to find something, but it's not the thing itself.

Example:
```c
int x = 10;        // Regular variable
int *ptr = &x;     // Pointer storing address of x
printf("%d", *ptr); // Outputs: 10
```

The `&` operator gets the address, and `*` dereferences (accesses the value at that address)."""
        else:
            return "A pointer stores a memory address. Use `&` to get an address, and `*` to access the value at that address."
    
    elif 'array' in lower:
        if difficulty > 0.6:
            return """Arrays in C are contiguous memory blocks. Array name is actually a pointer to the first element.

```c
int arr[5] = {1, 2, 3, 4, 5};
// arr is equivalent to &arr[0]
// arr[i] is equivalent to *(arr + i)
```

This pointer-array relationship is fundamental to understanding C memory model."""
        else:
            return """An array is a collection of elements of the same type stored in consecutive memory locations.

```c
int numbers[5] = {10, 20, 30, 40, 50};
printf("%d", numbers[0]);  // Access first element: 10
```

Arrays use zero-based indexing (first element is at index 0)."""
    
    elif 'loop' in lower or 'for' in lower or 'while' in lower:
        return """C has three main loop types:

1. **for loop** (when you know iteration count):
```c
for(int i = 0; i < 5; i++) {
    printf("%d ", i);
}
```

2. **while loop** (condition-based):
```c
int i = 0;
while(i < 5) {
    printf("%d ", i);
    i++;
}
```

3. **do-while loop** (executes at least once):
```c
int i = 0;
do {
    printf("%d ", i);
    i++;
} while(i < 5);
```"""
    
    elif 'function' in lower:
        return """Functions in C help organize code into reusable blocks.

Basic structure:
```c
return_type function_name(parameters) {
    // function body
    return value;
}
```

Example:
```c
int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);  // result = 8
    return 0;
}
```"""
    
    elif 'printf' in lower or 'scanf' in lower:
        return """**printf** outputs to console, **scanf** reads input:

```c
#include <stdio.h>

int main() {
    int age;
    
    printf("Enter your age: ");
    scanf("%d", &age);  // Note the & for address
    
    printf("You are %d years old\\n", age);
    return 0;
}
```

Common format specifiers:
- %d (int), %f (float), %c (char), %s (string)"""
    
    # Generic response
    difficulty_text = "Let me break this down simply" if difficulty < 0.4 else "Here's how this works" if difficulty < 0.7 else "From a technical perspective"
    return f"""{difficulty_text}: {user_message if '?' in user_message else "That's an important C programming concept."}

{" I'll provide a detailed explanation with examples." if support > 0.6 else "Think about the fundamental concepts of how C manages memory and data types."}

Would you like me to explain a specific aspect in more detail?"""

# ============================================================================
# LEARNING TRACKER
# ============================================================================
def check_milestones(state, turn):
    """Check if student has reached any milestones"""
    if state['k'] >= 0.3 and not any(m['type'] == 'first_understanding' for m in st.session_state.milestones):
        st.session_state.milestones.append({
            'type': 'first_understanding',
            'turn': turn,
            'message': '🎯 First signs of C programming understanding!',
            'time': datetime.now().strftime("%H:%M")
        })
    
    if state['c'] >= 0.7 and not any(m['type'] == 'confident' for m in st.session_state.milestones):
        st.session_state.milestones.append({
            'type': 'confident',
            'turn': turn,
            'message': '💪 Showing strong confidence!',
            'time': datetime.now().strftime("%H:%M")
        })
    
    if state['k'] >= 0.6 and not any(m['type'] == 'advanced' for m in st.session_state.milestones):
        st.session_state.milestones.append({
            'type': 'advanced',
            'turn': turn,
            'message': '🚀 Engaging with advanced concepts!',
            'time': datetime.now().strftime("%H:%M")
        })
    
    if state['m'] >= 0.8 and not any(m['type'] == 'engaged' for m in st.session_state.milestones):
        st.session_state.milestones.append({
            'type': 'engaged',
            'turn': turn,
            'message': '⚡ Highly engaged with learning!',
            'time': datetime.now().strftime("%H:%M")
        })

def get_learning_level(k):
    """Get descriptive learning level"""
    if k < 0.2:
        return 'Novice'
    elif k < 0.4:
        return 'Beginner'
    elif k < 0.6:
        return 'Intermediate'
    elif k < 0.8:
        return 'Advanced'
    else:
        return 'Expert'

# ============================================================================
# MAIN APP
# ============================================================================
def main():
    # Login/Authentication
    if st.session_state.current_user is None:
        st.markdown('<div class="main-header">💻 Adaptive C Programming Tutor</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Continuous Control Policy System for Learning C Basics</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👋 Enter your username to start your personalized C programming journey!")
            username = st.text_input("Username", placeholder="Enter your username", key="login_username")
            
            if st.button("🚀 Start Learning", use_container_width=True):
                if username.strip():
                    st.session_state.current_user = {
                        'username': username.strip(),
                        'created_at': datetime.now().isoformat(),
                        'session_count': 1
                    }
                    st.rerun()
                else:
                    st.error("Please enter a username")
        
        # Information section
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 Adaptive Learning")
            st.write("System adjusts difficulty, pacing, and support based on your responses")
        with col2:
            st.markdown("### 📊 Progress Tracking")
            st.write("Monitor your knowledge growth, confidence, and engagement over time")
        with col3:
            st.markdown("### 💡 C Programming Basics")
            st.write("Learn pointers, arrays, loops, functions, and more")
        
        return
    
    # Main Application (after login)
    user = st.session_state.current_user
    
    # Sidebar
    with st.sidebar:
        st.markdown(f"### 👤 {user['username']}")
        st.markdown("---")
        
        # Demo Mode Toggle
        st.session_state.demo_mode = st.toggle("🎮 Demo Mode", value=st.session_state.demo_mode, 
                                               help="Use simulated responses (no API key needed)")
        
        st.markdown("---")
        
        # Current State Display
        if st.session_state.learning_history:
            st.markdown("### 📊 Current State")
            latest = st.session_state.learning_history[-1]
            
            st.metric("Knowledge (K)", f"{latest['k']:.2f}")
            st.progress(latest['k'])
            
            st.metric("Confidence (C)", f"{latest['c']:.2f}")
            st.progress(latest['c'])
            
            st.metric("Engagement (M)", f"{latest['m']:.2f}")
            st.progress(latest['m'])
            
            st.caption(f"Error Type: {latest['e']}")
            
            # Learning Level
            level = get_learning_level(latest['k'])
            st.markdown(f"**Level:** {level}")
        
        st.markdown("---")
        
        # Milestones
        if st.session_state.milestones:
            st.markdown("### 🏆 Milestones")
            for milestone in st.session_state.milestones:
                st.success(f"{milestone['message']}\n\n*Turn {milestone['turn']} • {milestone['time']}*")
        
        st.markdown("---")
        
        # Actions
        if st.button("🔄 New Session", use_container_width=True):
            st.session_state.messages = []
            st.session_state.learning_history = []
            st.session_state.milestones = []
            st.session_state.turn_counter = 0
            st.rerun()
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.current_user = None
            st.session_state.messages = []
            st.session_state.learning_history = []
            st.session_state.milestones = []
            st.session_state.turn_counter = 0
            st.rerun()
        
        # Shareable Link
        st.markdown("---")
        st.markdown("### 🔗 Your Personal Link")
        base_url = "your-app-url.streamlit.app"  # Replace with actual URL after deployment
        personal_link = f"https://{base_url}?user={user['username']}"
        st.code(personal_link, language=None)
        st.caption("Share this link to access your session")
    
    # Main Content
    st.markdown('<div class="main-header">💻 C Programming Tutor</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">Welcome back, {user["username"]}!</div>', unsafe_allow_html=True)
    
    # Progress Chart
    if len(st.session_state.learning_history) > 1:
        with st.expander("📈 Learning Trajectory", expanded=False):
            turns = [h['turn'] for h in st.session_state.learning_history]
            knowledge = [h['k'] * 100 for h in st.session_state.learning_history]
            confidence = [h['c'] * 100 for h in st.session_state.learning_history]
            engagement = [h['m'] * 100 for h in st.session_state.learning_history]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=turns, y=knowledge, mode='lines+markers', name='Knowledge', line=dict(color='#3B82F6', width=3)))
            fig.add_trace(go.Scatter(x=turns, y=confidence, mode='lines+markers', name='Confidence', line=dict(color='#06B6D4', width=3)))
            fig.add_trace(go.Scatter(x=turns, y=engagement, mode='lines+markers', name='Engagement', line=dict(color='#EC4899', width=3)))
            
            fig.update_layout(
                title="Learning Progress Over Time",
                xaxis_title="Turn",
                yaxis_title="Score (%)",
                yaxis=dict(range=[0, 100]),
                template="plotly_dark",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Progress Summary
            col1, col2, col3 = st.columns(3)
            first = st.session_state.learning_history[0]
            latest = st.session_state.learning_history[-1]
            
            with col1:
                growth = (latest['k'] - first['k']) * 100
                st.metric("Knowledge Growth", f"{growth:+.0f}%")
            with col2:
                growth = (latest['c'] - first['c']) * 100
                st.metric("Confidence Growth", f"{growth:+.0f}%")
            with col3:
                st.metric("Total Turns", st.session_state.turn_counter)
    
    # Chat Interface
    st.markdown("---")
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        if not st.session_state.messages:
            st.info("""
            👋 **Welcome to your C Programming learning session!**
            
            Try asking:
            - "What are pointers in C?"
            - "How do I use arrays?"
            - "Explain for loops to me"
            - "I'm confused about malloc and free"
            """)
        
        for msg in st.session_state.messages:
            if msg['role'] == 'student':
                st.markdown(f'<div class="student-message"><b>You:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="tutor-message"><b>Tutor:</b><br>{msg["content"]}</div>', unsafe_allow_html=True)
    
    # Input area
    user_input = st.text_area("Ask a question about C programming:", 
                              placeholder="e.g., 'How do pointers work?'",
                              height=100,
                              key="user_input")
    
    col1, col2 = st.columns([6, 1])
    with col2:
        send_button = st.button("Send 📤", use_container_width=True)
    
    if send_button and user_input.strip():
        # Initialize components
        state_estimator = StateEstimator()
        policy_controller = PolicyController()
        prompt_generator = PromptGenerator()
        
        # Add user message
        st.session_state.messages.append({
            'role': 'student',
            'content': user_input,
            'timestamp': datetime.now().isoformat()
        })
        
        # Estimate state
        st.session_state.turn_counter += 1
        state = state_estimator.estimate(user_input)
        state['turn'] = st.session_state.turn_counter
        st.session_state.learning_history.append(state)
        
        # Check milestones
        check_milestones(state, st.session_state.turn_counter)
        
        # Select action
        action = policy_controller.select_action(state)
        
        # Generate response
        with st.spinner("Tutor is thinking..."):
            if st.session_state.demo_mode:
                time.sleep(0.8)  # Simulate thinking
                response = generate_demo_response(state, action, user_input)
            else:
                # Real API call (requires ANTHROPIC_API_KEY in secrets)
                try:
                    client = anthropic.Anthropic(api_key=st.secrets.get("ANTHROPIC_API_KEY", ""))
                    prompt = prompt_generator.build_prompt(action, user_input, st.session_state.messages, state)
                    
                    message = client.messages.create(
                        model="claude-sonnet-4-20250514",
                        max_tokens=1000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    
                    response = message.content[0].text
                except Exception as e:
                    response = f"Error: {str(e)}. Please enable Demo Mode or add your API key to secrets."
        
        # Add tutor response
        st.session_state.messages.append({
            'role': 'tutor',
            'content': response,
            'timestamp': datetime.now().isoformat()
        })
        
        st.rerun()

if __name__ == "__main__":
    main()