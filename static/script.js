// Dark mode toggle slider functionality
const darkModeToggle = document.getElementById('dark-mode-toggle');
const modeText = document.getElementById('mode-text');

darkModeToggle.addEventListener('change', () => {
  document.documentElement.classList.toggle('dark-mode', darkModeToggle.checked);
  // Update the mode text accordingly.
  modeText.textContent = darkModeToggle.checked ? "Dark Mode" : "Light Mode";
});


// Listener for the chat form submission
document.getElementById('chat-form').addEventListener('submit', async (e) => {
  e.preventDefault();
  const input = document.getElementById('user-input');
  const userInput = input.value.trim();
  if (!userInput) return;
  addMessage(userInput, false);
  input.value = '';

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: userInput })
    });
    const { response } = await res.json();
    addMessage(response, true);
  } catch (err) {
    console.error(err);
    addMessage("Something went wrong. Please try again later.", true);
  }
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

// Event delegation for toggling collapsible details.
document.getElementById('chat-container').addEventListener('click', (e) => {
  if (e.target && e.target.classList.contains('toggle-btn')) {
    const details = e.target.nextElementSibling;
    if (details) {
      details.style.display = (details.style.display === "none" || details.style.display === "") ? "block" : "none";
    }
  }
});
