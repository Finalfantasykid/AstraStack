var $_GET = {page: "home"};

document.location.search.replace(/\??(?:([^=]+)=([^&]*)&?)/g, function () {
    function decode(s) {
        return decodeURIComponent(s.split("+").join(" "));
    }

    $_GET[decode(arguments[1])] = decode(arguments[2]);
});

function resizeBody(){
    var minHeight = $(window).height() - ($("#footer_container").outerHeight(true) + 
                                          $("#nav_container").outerHeight(true) +
                                          $("#header_container").outerHeight(true));
    $("#body_container").css("min-height", minHeight-16);
}

$(document).ready(function(){
    $("a[data-page=" + $_GET.page + "]").addClass("selected");
    $.get($_GET.page + ".html", function(response){
        $("#body").html(response);
        if($_GET.page != "home"){
            $("title").text($("div#body h1").text() + " - AstraStack");
        }
    });
    resizeBody();
});
$(window).resize(resizeBody);
