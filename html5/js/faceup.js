

if (XMLHttpRequest.prototype.sendAsBinary === undefined) {
  XMLHttpRequest.prototype.sendAsBinary = function(string) {
    var bytes = Array.prototype.map.call(string, function(c) {
      return c.charCodeAt(0) & 0xff;
    });
    this.send(new Uint8Array(bytes).buffer);
  };
}


function postCanvasToURL() {
	document.getElementById("submit").style.visibility = "hidden"
	var type = "image/jpeg"
	var data = document.getElementById("can").toDataURL(type);
	if ( ! data ) return
	data = data.replace('data:' + type + ';base64,', '');

	var xhr = new XMLHttpRequest();
	xhr.open('POST', "/", true);
	xhr.onreadystatechange = function()	{
		if ( xhr.readyState < 4 ) {
			document.getElementById("compout").innerHTML = xhr.readyState + " " + xhr.status + " " + "posting ..<br>"
		} else {
			document.getElementById("compres").innerHTML = xhr.responseText
		}
	}
	var  n = document.getElementById("n").value
	var fn = document.getElementById("f").value
	var boundary = 'ohaiimaboundary';
	xhr.setRequestHeader('Content-Type', 'multipart/form-data; boundary=' + boundary);
	xhr.sendAsBinary([
		'--' + boundary,
		'Content-Disposition: form-data; name="n"',
		'',
		n,
		'--' + boundary,
		'Content-Disposition: form-data; name="f"; filename="' + fn + '"',
		'Content-Type: ' + type,
		'',
		atob(data),
		'--' + boundary + "--", 
		''
	].join('\r\n'));
}

//~ function detectim() {
	//~ if ( document.getElementById("u").value )
		//~ document.getElementById("download").style.visibility = "visible"
//~ }

function loadim() {
	var co = document.getElementById("compout")
	co.innerHTML = "processing.."	
	document.getElementById("n").value = ""
	document.getElementById("n").style.visibility = "hidden"
	document.getElementById("submit").style.visibility = "hidden"
	//~ document.getElementById("download").style.visibility = "hidden"
	var face_ctx = document.getElementById("can").getContext("2d");
	face_ctx.fillRect(0,0,90,90)
	
	var image = new Image();
	//~ if ( islocal ) {
	var matchim = document.getElementById("f")
	//~ image.src = matchim.files[0].getAsDataURL()
	oFReader = new FileReader()
	oFReader.onload = function (oFREvent) {
		image.src = oFREvent.target.result;
		if ( !image.src )  { 
			co.innerHTML = "invalid image src"
			return false
		}
		return true
	};
    var oFile = matchim.files[0];
    oFReader.readAsDataURL(oFile);
	//~ } else {
		//~ var matchurl = document.getElementById("u").value
		//~ image.src = matchurl
		//~ image.tagName = "img"
	//~ }

	co.innerHTML = "loading.."	
	image.onload = function () {
		co.innerHTML = "detecting.."
		var comp = []
		try {
			var aa = ccv.pre(image)
			var bb = ccv.grayscale(aa)
			comp = ccv.detect_objects({ "canvas" : bb,
											"cascade" : cascade,
											"interval" : 5,
											"min_neighbors" : 1 });
		} catch(e) {
			co.innerHTML = "face detect failed." + e
			return false
		}
										
		co.innerHTML = comp.length  + " faces found.<br>"
		if( comp.length  > 0 ) {
			var x = Math.ceil(comp[0].x)
			var y = Math.ceil(comp[0].y)
			var w = Math.ceil(comp[0].width)
			var h = Math.ceil(comp[0].height)
			if ( h + h/4 < 90 ) {
				h += h/4
			}
			co.innerHTML += "[" + x + "," + y + "," + w + "," + h + "]<br>\n"
			face_ctx.drawImage(image, x,y,w,h, 0,0,90,90);
			document.getElementById("submit").style.visibility = "visible"
			document.getElementById("n").style.visibility = "visible"
		}
		return true;
	}	
	image.onerror = function () {
		co.innerHTML = "image load failed." + image.src.substring(0,20)
	}
	return true;
}
