/* ------------------------------------------------------------------
   Real‑time face + weapon overlay & gallery – FULL JS (smooth boxes)
   ------------------------------------------------------------------*/

// ---------- DOM elem ----------
const video        = document.getElementById("video");
const overlay      = document.getElementById("overlay");
const resultP      = document.getElementById("result");
const capturesDiv  = document.getElementById("captures");
const ctx          = overlay.getContext("2d");

// ---------- globals ----------
let containerToUpdate   = null;
let originalNameToReplace = null;
const nameOverrides  = new Map();
const shownCaptures  = new Set();
const shownWeaponUIDs = new Set();

const scaleX = 1;
const scaleY = 1;

// -------------- SMOOTH ANIMATION SETTINGS --------------
const SMOOTH = 1;      // 0 = instant, 1 = nema pomaka
let   prevBoxes = [];     // spremamo kutije iz prošlog frame‑a

// ---------- WebSocket ----------
const ws = new WebSocket("ws://127.0.0.1:8000/ws");

ws.onmessage = (event) => {
    const data = JSON.parse(event.data);

    const boxes = [ ...(data.face_boxes ?? []), ...(data.weapon_boxes ?? []) ];
    showBoundingBoxes(boxes);

    if (data.face_captures) {
        showCaptures(data.face_captures);
    }

    if (data.faces) {
        resultP.innerText = "Faces: " + data.faces.join(", ");
    }
};

// ---------- video → websocket ----------
navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream;
        const canvas = document.createElement("canvas");
        const cctx   = canvas.getContext("2d");

        setInterval(() => {
            if (!video.videoWidth || !video.videoHeight) {
                return;
            }
            if (ws.readyState === WebSocket.OPEN) {
                canvas.width  = video.videoWidth  / scaleX;
                canvas.height = video.videoHeight / scaleY;
                cctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL("image/jpeg", 0.9);
                ws.send(dataURL);
            }
        }, 500);
    })
    .catch(err => {
        console.error("getUserMedia error", err);
    });

// ---------- overlay helpers ----------
function getBoundingBoxInfoLines(box) {
    if (box.weapon) {
        return [
            `Weapon: ${box.label} (${(box.confidence * 100).toFixed(0)}%)`
        ];
    }
    return [
        `Criminal record: ${box.criminal_record || "N/A"}`,
        `Nationality: ${box.nationality || "N/A"}`,
        `Age: ${box.age || "N/A"}`,
        `Name: ${box.name} ${box.surname || ""}`
    ];
}

// linearna interpolacija
const lerp = (a, b, t) => a + (b - a) * t;

function smoothBox(prev, curr) {
    if (!prev) {
        return curr;
    }
    return {
        ...curr,
        top:    lerp(prev.top,    curr.top,    SMOOTH),
        right:  lerp(prev.right,  curr.right,  SMOOTH),
        bottom: lerp(prev.bottom, curr.bottom, SMOOTH),
        left:   lerp(prev.left,   curr.left,   SMOOTH)
    };
}

function showBoundingBox(ctx, box) {
    let color;
    if (box.weapon) {
        color = "red";
    } else if (box.criminal_record) {
        color = "red";
    } else {
        color = "lime";
    }

    ctx.strokeStyle = color;
    ctx.lineWidth   = 2;
    ctx.strokeRect(box.left * scaleX, box.top * scaleY,
                   (box.right - box.left) * scaleX,
                   (box.bottom - box.top) * scaleY);

    ctx.font      = "16px Arial";
    ctx.fillStyle = color;
    const info = getBoundingBoxInfoLines(box);
    info.forEach((line, i) => {
        ctx.fillText(line, box.left * scaleX + 5, box.top * scaleY - 5 - i * 18);
    });
}

function showBoundingBoxes(boxes) {
    ctx.clearRect(0, 0, overlay.width, overlay.height);

    // zagladi svaku kutiju prema prethodnoj
    const smooth = boxes.map((curr, idx) => smoothBox(prevBoxes[idx], curr));
    smooth.forEach(b => showBoundingBox(ctx, b));
    prevBoxes = smooth;   // spremi za idući frame
}


// ---------- captures / gallery ----------
function addCapture(capture) {
    const idKey = `${capture.image}-${capture.time}`;
    if (shownCaptures.has(idKey)) return;
    shownCaptures.add(idKey);

    const container = document.createElement("div");
    container.className = "capture-container";
	if (capture.criminal_record && capture.criminal_record.trim() !== "") {
		container.classList.add("capture-danger");
	} else if (capture.weapon) {
		container.classList.add("capture-weapon");
	}
	
	//  Ako je “unknown …” – dodaj gumb za brisanje
	/*if (capture.name.startsWith("unknown ")) {
		addRemoveButton(container, idKey);
	}*/

    const img = document.createElement("img");
    img.src = capture.image;
    img.width = 240;

    const caption = document.createElement("div");
    caption.className = "capture-caption";
    caption.innerText = `${capture.name} ${capture.surname}\n${formatDateHR(capture.time)}`;

    container.appendChild(img);
    container.appendChild(caption);
    capturesDiv.appendChild(container);

    // weapon capture – bez gumba
    /*if (capture.weapon) {
		if (!shownWeaponUIDs.has(capture.uid)) {
			shownWeaponUIDs.add(capture.uid);
		}
		return;
	}*/

    // ---- lice: dodaj gumbe ----
    const originalName = capture.name;
    const displayName  = nameOverrides.get(originalName) || originalName;
    const knownPerson  = nameOverrides.has(originalName) || !originalName.startsWith("unknown ");

    if (knownPerson) {
        createEditButton(container, capture, originalName, displayName, capture.id || originalName);
    } else {
        createAddButton(container, capture, originalName);
    }
}

function showCaptures(captures) {
    captures.forEach(addCapture);
}

function createEditButton(container, capture, originalName, displayName, id) {
	const editBtn = document.createElement("button");
	editBtn.innerText = "Uredi osobu";
	editBtn.className = "save-person-button";
	editBtn.onclick = () => {
		containerToUpdate = container;
		originalNameToReplace = originalName;
		fetch(`/get_person/${id}`)
			.then(resp => resp.json())
			.then(data => {
				document.getElementById("popup-form").style.display = "block";
				document.getElementById("popupImageData").value = capture.image;
				document.getElementById("personId").value = data.id || "";
				document.querySelector("input[name='name']").value = data.name || "";
				document.querySelector("input[name='surname']").value = data.surname || "";
				document.querySelector("input[name='age']").value = data.age || "";
				document.querySelector("input[name='nationality']").value = data.nationality || "";
				document.querySelector("input[name='criminal_record']").value = data.criminal_record || "";
			});
	};
	container.appendChild(editBtn);

	// Dodaj gumb za prikaz slike
	createImageViewIcon(container, id);
}

function createAddButton(container, capture, originalName) {
	const saveBtn = document.createElement("button");
	saveBtn.innerText = "Spremi osobu";
	saveBtn.className = "save-person-button";
	saveBtn.onclick = () => {
		containerToUpdate = container;
		originalNameToReplace = originalName;
		const form = document.getElementById("personForm");
		form.reset();
		//document.getElementById("personId").value = "";
		document.getElementById("popupImageData").value = capture.image;
		document.getElementById("popup-form").style.display = "block";
	};
	container.appendChild(saveBtn);
}

navigator.mediaDevices.getUserMedia({ video: true })
    .then(stream => {
        video.srcObject = stream; // prikazuje video

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");

        setInterval(() => {
            if (video.videoWidth === 0 || video.videoHeight === 0) {
				return;
			}
            if (ws.readyState === WebSocket.OPEN) {
				canvas.width = video.videoWidth / scaleX;
				canvas.height = video.videoHeight / scaleY;
				ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
                const dataURL = canvas.toDataURL("image/jpeg", 0.9); // manja kvaliteta = brže
                ws.send(dataURL);
            }
        }, 500); // svakih 500 ms
    })
    .catch(err => {
        console.error("Error accessing webcam: ", err);
    });

document.getElementById("personForm").addEventListener("submit", function (e) {
    e.preventDefault();
    const formData = new FormData(this);
    const id = formData.get("id");
    const url = id ? "/update_person" : "/save_unknown_person";

    fetch(url, {
        method: "POST",
        body: formData
    })
    .then(resp => resp.json())
	.then(data => {
		if (id && containerToUpdate) {
			saveExistingPerson(formData, containerToUpdate);
		}
		if (!id && data.id && containerToUpdate && originalNameToReplace) {
			saveUnknownPerson(formData, containerToUpdate, data.id);
		}
		document.getElementById("popup-form").style.display = "none";
		showInfo(id ? "Osoba ažurirana!" : "Osoba spremljena!");
	})
	.catch(err => alert("Greška pri spremanju: " + err));
});

function saveExistingPerson(formData, containerToUpdate) {
    const newName     = formData.get("name");
    const newSurname  = formData.get("surname") || "";
    const oldDisplay  = containerToUpdate.querySelector("div")?.innerText.split("\n")[0];

    shownCaptures.add(`${newName} ${newSurname}`);

    if (oldDisplay && oldDisplay !== `${newName} ${newSurname}`) {
        // prepiši stari mapping → novi (ime + prezime)
        for (const [originalName, mappedName] of nameOverrides.entries()) {
            if (mappedName === oldDisplay) {
                nameOverrides.set(originalName, `${newName} ${newSurname}`);
                break;
            }
        }
        nameOverrides.set(originalNameToReplace, `${newName} ${newSurname}`);
    }

    // osvježi caption
    const caption = containerToUpdate.querySelector("div");
    if (caption) {
        const timePart = caption.innerText.split("\n")[1];
        caption.innerText = `${newName} ${newSurname}\n${timePart}`;
    }

    containerToUpdate     = null;
    originalNameToReplace = null;
}

function saveUnknownPerson(formData, containerToUpdate, id) {
    const newName    = formData.get("name");
    const newSurname = formData.get("surname") || "";

    // mapiraj privremeni "unknown …" → "Ime Prezime"
    nameOverrides.set(originalNameToReplace, `${newName} ${newSurname}`);
    shownCaptures.add(`${newName} ${newSurname}`);

    // zamijeni gumb i dodaj “info” ikonu
    const oldBtn = containerToUpdate.querySelector("button");
    if (oldBtn) oldBtn.remove();

    const editBtn = document.createElement("button");
    editBtn.innerText = "Uredi osobu";
    editBtn.className = "save-person-button";
    editBtn.onclick = () => {
		fetch(`/get_person/${id}`)
			.then(resp => resp.json())
			.then(data => {
				document.getElementById("popup-form").style.display = "block";
				document.getElementById("popupImageData").value = containerToUpdate.querySelector("img").src;
				document.getElementById("personId").value = data.id || "";
				document.querySelector("input[name='name']").value = data.name || "";
				document.querySelector("input[name='surname']").value = data.surname || "";
				document.querySelector("input[name='age']").value = data.age || "";
				document.querySelector("input[name='nationality']").value = data.nationality || "";
				document.querySelector("input[name='criminal_record']").value = data.criminal_record || "";
			});
	};
	document.getElementById("personId").value = id;
    containerToUpdate.appendChild(editBtn);
    createImageViewIcon(containerToUpdate, id);

    // novi caption
    const caption = containerToUpdate.querySelector("div");
    if (caption) {
        caption.innerText = `${newName} ${newSurname}\n${formatDateHR(new Date().toISOString())}`;
    }

    containerToUpdate     = null;
    originalNameToReplace = null;
}

function showInfo(content, duration = 3000) {
	const messageDiv = document.getElementById("status-message");
    messageDiv.innerText = content;
    messageDiv.classList.remove("hidden");

    setTimeout(() => {
        messageDiv.classList.add("hidden");
    }, duration);
}

function createImageViewIcon(container, personId) {
	const viewBtn = document.createElement("button");
	viewBtn.title = "Prikaži referentnu sliku";
	viewBtn.className = "icon-button";

	const iconImg = document.createElement("img");
	iconImg.src = "/static/info.svg";
	iconImg.alt = "Prikaži sliku";
	iconImg.className = "icon-image";
	viewBtn.appendChild(iconImg);

	viewBtn.onclick = () => {
		document.getElementById("popup-person-image").src = `/person_image/${personId}`;
		document.getElementById("image-popup").classList.remove("hidden");
	};
	container.appendChild(viewBtn);
}

// --------------------------------------------------------
//  Doda “×” gumb koji briše capture samo s front-enda
// --------------------------------------------------------
function addRemoveButton(container, idKey) {
    const btn = document.createElement("button");
    btn.className = "remove-btn";
    btn.textContent = "×";
    btn.onclick = () => {
        container.remove();           // makni iz DOM-a
        shownCaptures.delete(idKey);  // dopusti ponovno dodavanje ako se opet pojavi
    };
    container.appendChild(btn);
}

// ----------------------------------------------------
// Pretvori 'YYYY-MM-DD HH:MM:SS'  ➜  '29.6.2025. 19:35:42'
// ----------------------------------------------------
function formatDateHR(ts) {
    // 1) “ISO-ify” – zamijeni razmak s “T” ako ga ima
    const iso = ts.includes(" ") ? ts.replace(" ", "T") : ts;
    const d   = new Date(iso);

    // 2) Ručno složi datum bez razmaka (dan.mjesec.godina.)
    const day   = d.getDate();          // 1-31
    const month = d.getMonth() + 1;     // 0-11  ➜ 1-12
    const year  = d.getFullYear();

    // 3) Vrijeme i dalje prepusti lokalnoj funkciji
    const time  = d.toLocaleTimeString("hr-HR");  // npr. 19:35:42

    return `${day}.${month}.${year}. ${time}`;
}




