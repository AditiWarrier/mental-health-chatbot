console.log("✅ script.js loaded");
document.addEventListener("DOMContentLoaded", function () {
  const userInput = document.getElementById("userInput");
  const sendButton = document.getElementById("sendButton");
  const chatBox = document.getElementById("chatbox");

  function scrollToBottom() {
    chatBox.scrollTop = chatBox.scrollHeight;
  }

  function addMessage(sender, text) {
    const wrapper = document.createElement("div");
    wrapper.classList.add("message", sender);
    const bubble = document.createElement("div");
    bubble.classList.add("bubble");
    bubble.innerText = text;
    wrapper.appendChild(bubble);
    chatBox.appendChild(wrapper);
    scrollToBottom();
  }

  async function sendMessage() {
    console.log("🟢 sendMessage() triggered");
    const userMessage = userInput.value.trim();
    if (!userMessage) return;

    addMessage("user", userMessage);
    userInput.value = "";

    try {
      const response = await fetch("/get", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ msg: userMessage }),
        credentials: "include" // important: send cookies so Flask sees session
      });

      console.log("fetch returned, status:", response.status);
      const text = await response.text();
      console.log("bot text:", text);
      addMessage("bot", text);
    } catch (error) {
      console.error("Fetch error:", error);
      addMessage("bot", "⚠️ Error connecting to Serene.");
    }
  }

  sendButton.addEventListener("click", sendMessage);
  userInput.addEventListener("keypress", (e) => {
    if (e.key === "Enter") sendMessage();
  });
});
