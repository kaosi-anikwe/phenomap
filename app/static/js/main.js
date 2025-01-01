// Show toasts
var toastElList = [].slice.call(document.querySelectorAll(".toast"));
var toastList = toastElList.map(function (toastEl) {
  return new bootstrap.Toast(toastEl, { delay: 5000 });
});
toastList.forEach((toast) => toast.show());

const flashMessage = (message, catetgory = "info", delay = 5000) => {
  const flashContainer = document.getElementById("flash-container");
  let newToast = document.createElement("div");
  newToast.setAttribute(
    "class",
    `toast align-items-center text-white bg-${catetgory} border-0`
  );
  newToast.innerHTML = `
            <div class="d-flex">
              <div class="toast-body">${message}</div>
                <button
                  type="button"
                  class="mx-2 close"
                  data-dismiss="toast"
                  aria-label="Close"
                >
                  <span aria-hidden="true">&times;</span>
                </button>
            </div>
            `;
  flashContainer.appendChild(newToast);
  newToast = new bootstrap.Toast(newToast, { delay: delay });
  newToast.show();
};

// Resend verification email
if (document.getElementById("resend-email")) {
  document
    .getElementById("resend-email")
    .addEventListener("click", async () => {
      try {
        let response = await fetch("/send-verification-email");
        if (response.ok) {
          flashMessage("Verification email sent successfully!", "success");
          return;
        }
        flashMessage(
          "Verification email failed to send. Please try again later.",
          "danger"
        );
      } catch (error) {
        console.error(error);
        flashMessage(
          "Verification email failed to send. Please try again later.",
          "danger"
        );
      }
    });
}

// Add Case
const addCase = async (verified = false) => {
  try {
    if (!verified) {
      flashMessage("Please verify your email before adding a case.");
      return;
    }
    let response = await fetch("/api/cases/update", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uid: "" }),
    });
    if (response.ok) {
      let data = await response.json();
      window.location.href = `/case/${data.uid}`;
    } else {
      console.log(await response.text());
      flashMessage(
        "There was an error. Refresh the page and try again.",
        "danger"
      );
    }
  } catch (error) {
    console.error(error);
    flashMessage(
      "Something went wrong. Please refresh the page and try again.",
      "danger"
    );
  }
};

$(document).ready(function () {
  // Preserve original dropdown functionality
  $("#profileDropdown").on("click", function (e) {
    e.preventDefault();
    $(this).parent().toggleClass("show");
    $(this).next(".dropdown-menu").toggleClass("show");
  });

  // Close dropdown when clicking outside
  $(document).on("click", function (e) {
    if (!$(e.target).closest(".nav-profile").length) {
      $(".nav-profile.show").removeClass("show");
      $(".dropdown-menu.show").removeClass("show");
    }
  });
});
