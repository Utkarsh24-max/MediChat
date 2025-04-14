// Dark mode toggle slider functionality
const darkModeToggle = document.getElementById('dark-mode-toggle');
const modeText = document.getElementById('mode-text');

darkModeToggle.addEventListener('change', () => {
  document.documentElement.classList.toggle('dark-mode', darkModeToggle.checked);
  // Update the mode text accordingly.
  modeText.textContent = darkModeToggle.checked ? "Dark Mode" : "Light Mode";
});

// Function to add a new message to the chat container.
function addMessage(text, isBot) {
  const chat = document.getElementById('chat-container');
  const msg = document.createElement('div');
  msg.className = `message ${isBot ? 'bot' : 'user'}`;
  
  if (isBot) {
    // Get the bot icon URL from the chat container's data attribute.
    const botIconUrl = chat.getAttribute('data-bot-icon');
    msg.innerHTML = `
      <img src="${botIconUrl}" alt="Bot Icon" class="bot-icon" />
      <div class="message-text">${text}</div>
    `;
  } else {
    msg.innerHTML = `<div class="message-text">${text}</div>`;
  }
  
  chat.appendChild(msg);
  chat.scrollTop = chat.scrollHeight;
}

// Listener for the chat form submission (with loading indicator).
document.getElementById('chat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('user-input');
  const userInput = input.value.trim();
  if (!userInput) return;
  
  // Append the user message.
  addMessage(userInput, false);
  input.value = '';
  
  // Create and append the loading indicator with jumping dots.
  const chat = document.getElementById('chat-container');
  const loadingElem = document.createElement('div');
  loadingElem.className = 'message bot loading';
  loadingElem.innerHTML = `
    <img src="${chat.getAttribute('data-bot-icon')}" alt="Bot Icon" class="bot-icon" />
    <div class="message-text">
      <span class="jumping-dots">
        <span class="dot-1"></span>
        <span class="dot-2"></span>
        <span class="dot-3"></span>
      </span>
    </div>
  `;
  chat.appendChild(loadingElem);
  chat.scrollTop = chat.scrollHeight;
  
  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userInput })
    });
    const { response } = await res.json();
    // Remove the loading indicator.
    loadingElem.remove();
    // Append the bot's response.
    addMessage(response, true);
  } catch (err) {
    console.error(err);
    loadingElem.remove();
    addMessage("Something went wrong. Please try again later.", true);
  }
});

// Event delegation for toggling collapsible details.
document.getElementById('chat-container').addEventListener('click', (e) => {
  if (e.target && e.target.classList.contains('toggle-btn')) {
    const details = e.target.nextElementSibling;
    if (details) {
      details.style.display = (details.style.display === "none" || details.style.display === "") ? "block" : "none";
    }
  }
});