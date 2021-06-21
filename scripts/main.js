var $_GET = {page: "home"};

document.location.search.replace(/\??(?:([^=]+)=([^&]*)&?)/g, function () {
    function decode(s) {
        return decodeURIComponent(s.split("+").join(" "));
    }

    $_GET[decode(arguments[1])] = decode(arguments[2]);
});

$(document).ready(function(){
    $("a[data-page=" + $_GET.page + "]").addClass("selected");
    $.get($_GET.page + ".html", function(response){
        $("#body").html(response);
    });
});
