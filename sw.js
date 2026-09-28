/* Karma Chat - online client. The app works via the server API only now,
   so there is nothing to cache: this worker just deletes any old caches
   left over from the earlier offline versions. */
self.addEventListener("install", (e) => e.waitUntil(self.skipWaiting()));
self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener("fetch", (e) => { /* network only */ });
