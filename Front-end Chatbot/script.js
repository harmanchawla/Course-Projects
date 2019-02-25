let bot = new RiveScript();
bot.sortReplies();

// let dictation = new responsiveVoice();
// let speech = new P5.Speech();
/*
 We need 4 references:
1. The div element where messages are displayed
2. Div element for bot BotReply
3. Div element for user input to be displayed
4. form
4. User text input
*/
const messageContainer = document.getElementById("conversation");
const botreply = document.getElementById('botreply');
const userinput = document.getElementById('userinput');
const form = document.querySelector('form');
const inputText = document.querySelector('.message');

const brains =["genericBrain.rive", "staticBrain.rive", "dynamicBrain.rive"];

var counter = 0;

bot.loadFile(brains).then(botReady).catch(botNotReady);

function botReady(){
  console.log('Bot Ready');
  bot.sortReplies();
}

function botNotReady(err){
//  alert('There was a problem while booting up the system: ' + err);
  console.log(err);
}

form.addEventListener('submit', (e) => {
  e.preventDefault();
  selfReply(inputText.value);
  inputText.value = '';
});

// This is the reply the bot generates
function botReply(message){
  console.log(message);
  var innerHTMLText1 = '<div class="message-container"> <img src="bot-avatar.png" style="width: 100%" /> <br /> <p class="chat-response-block" id="bot-response" style="word-break: keep-all;">';
  var innerHTMLText2 = message;
  var innerHTMLText3 = '<br /> <span class="time-left" id="time1">';
  var innerHTMLText4 = getTimeHM();
  var innerHTMLText5 = '</span> </p> </div>';

  const toggleSwitch = document.getElementById('voiceSwitch').checked;

  if (toggleSwitch == true) {
    var voiceResponse = message.split(':')[0];
    responsiveVoice.speak(voiceResponse, "UK English Female");
  }

  var botStitchedReply = innerHTMLText1 + innerHTMLText2 + innerHTMLText3 + innerHTMLText4 + innerHTMLText5;
  if (counter%2 === 1) {
    userinput.innerHTML += botStitchedReply;
    counter = counter + 1;
  }
  else {
    botreply.innerHTML += botStitchedReply;
    counter = counter + 1;
  }
  location.href = '#edge';
  inputText.value = '';
  document.getElementById('user-text-input').focus();
}

// Display what the user has entered
function selfReply(message){
  console.log(message);
  var innerHTMLText1 = '<div class="message-container"> <img src="person-avatar.png" alt="Bot-Avatar" style="width: 100%" class="right" /> <p class="chat-input-block" id="user-response">';
  var innerHTMLText2 = message;
  var innerHTMLText3 = '<br/ ><span class="time-right" id="time2">';
  var innerHTMLText4 = getTimeHM();
  var innerHTMLText5 = '</span> </p> </div>';
  var userStitchedReply = innerHTMLText1 + innerHTMLText2 + innerHTMLText3 + innerHTMLText4 + innerHTMLText5;

  if (counter%2 === 0) {
    userinput.innerHTML += userStitchedReply;
    counter = counter + 1;
  }
  else {
    botreply.innerHTML += userStitchedReply;
    counter = counter + 1;
  }
  location.href = '#edge';
  bot.reply("local-user", message).then(function(reply) {
    botReply(reply);
 });
}

function getTimeHM() {
    var d = new Date(),
    minutes = d.getMinutes().toString().length == 1 ? '0'+d.getMinutes() : d.getMinutes(),
    hours = d.getHours().toString().length == 1 ? '0'+d.getHours() : d.getHours(),
    ampm = d.getHours() >= 12 ? 'PM' : 'AM';
    if (hours > 12) {
      hours = hours - 12;
    }
    return hours+ ':' +minutes+' '+ampm;
}
