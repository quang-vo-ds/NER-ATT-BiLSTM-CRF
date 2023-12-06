$(function(){

    
    $('#sentence_submit').on("click", function(){
		
		var sentence = $('#input_sentence').val();
		
				
		if (sentence){
                chrome.runtime.sendMessage(
					{input_sentence: sentence},
					function(response) {
						result = response.farewell;
						alert(result.tag);
						
						var notifOptions = {
                        type: "basic",
                        iconUrl: "icon48.png",
                        title: "Biomedial Name Entity Recognition",
                        message: result.tag
						};
						
						chrome.notifications.create('TagNotif', notifOptions);
						
					});
		}
			
			
		$('#keyword').val('');
		
    });
});