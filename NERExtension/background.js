var serverhost = 'http://127.0.0.1:8000';

	chrome.runtime.onMessage.addListener(
		function(request, sender, sendResponse) {
		  
			  
			var url = serverhost + '/ner/get_tags/?input_sentence='+ encodeURIComponent(request.input_sentence) ;
			
			console.log(url);
			
			//var url = "http://127.0.0.1:8000/ner/get_tags/?input_sentence='Clustering of missense mutations in the ataxia-telangiectasia gene in a sporadic T-cell leukaemia.'"	
			fetch(url)
			.then(response => response.json())
			.then(response => sendResponse({farewell: response}))
			.catch(error => console.log(error))
				
			return true;  // Will respond asynchronously.
		  
	});