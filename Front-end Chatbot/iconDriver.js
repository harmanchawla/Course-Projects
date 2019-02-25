function hideButtons() {
	var images = document.getElementById("chat-images");
	var chat = document.getElementById("chat-window")
  	images.style.display = "none";
  	chat.style.display = "in-line";
} 

function isOnline() {
	var online = navigator.onLine;
	if (online != true) {
		 alert("You don't seem to be online. Please check your internet connection.");
	}
	else {
		hideButtons()
	}
}